import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, PretrainedConfig
from transformers.trainer_utils import denumpify_detensorize, has_length
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


import os
import math
import wandb
import numpy as np
from pathlib import Path
from functools import partial
from typing import Any, Dict, Optional, Callable, Tuple, Union, List
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from models.softvq_loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, ReconstructionLoss, hinge_gen_loss
from models.softvq_vae import SoftVQConfig, SoftVQModel
from models.discriminators import MultiResolutionDiscriminator
from utils.diff_aug import DiffAugment

# MelSpec -> rec_loss

@dataclass
class GANTrainingArguments(TrainingArguments):
    n_critic: int = field(default=1, metadata={"help": "判别器每次更新的步数"})
    gp_lambda: float = field(default=10.0, metadata={"help": "WGAN-GP 的梯度惩罚系数"})
    ema: bool = field(default=False, metadata={"help": "是否对生成器做 EMA"})
    ema_decay: float = field(default=0.999, metadata={"help": "EMA 衰减系数"})
    use_gp: bool = field(default=False, metadata={"help": "是否使用 WGAN-GP"})
    loss_type: str = field(default="hinge", metadata={"help": "hinge|wgan|bce"})

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert n in self.shadow
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])


# ===========================
#  Helper: 判别器输出统一
# ===========================
def _unpack_disc_out(out):
    if isinstance(out, tuple):
        logits, feats = out
    else:
        logits, feats = out, None
    return logits, feats

def gradient_penalty(discriminator, real, fake, gp_lambda=10.0):
    """WGAN-GP"""
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device, dtype=real.dtype)
    interpolates = epsilon * real + (1 - epsilon) * fake
    interpolates.requires_grad_(True)

    disc_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(disc_interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gp


class SoftVQ_GANTrainer(Trainer):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        noise_sampler: Optional[Callable[..., torch.Tensor]],  # 可为 None
        train_args,  # GANTrainingArguments
        audio_param: Optional[Dict[str, Any]] = None,
        recons_loss: str = 'l1',
        optimizers: str = None,
        scheduler: str = 'linear',
        max_steps: Optional[int] = None,
        use_diff_aug: bool = False,
        rec_weight: float = 1.0,
        # device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(model=generator, args=train_args, **kwargs)
        self.args = train_args

        self.audio_param = audio_param if audio_param is not None else {}

        self.gen = generator
        self.disc = discriminator
        self.noise_sampler = noise_sampler  # 对你来说可能 None

        self.optimizers = optimizers
        self.max_steps = max_steps
        self.lr_scheduler_type = scheduler

        self.use_diff_aug = use_diff_aug

        self.gen_loss  = GeneratorLoss()
        self.disc_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.rec_loss = ReconstructionLoss(loss_type=recons_loss)
        self.multi_res_disc = MultiResolutionDiscriminator()
        self.multi_res_disc.to(train_args.device)
        self.gen_adv_loss = hinge_gen_loss

        # loss_weight
        self.rec_weight = rec_weight

        self._critic_steps = 0
        self._ema = EMA(self.gen, train_args.ema_decay) if getattr(train_args, "ema", False) else None

        self._phase = "D"  # 当前阶段：D 或 G

         # ---- mel coeff 衰减相关 ----
        self.base_mel_coeff   = getattr(train_args, "mel_coeff", 45.0)
        self.mel_loss_coeff   = self.base_mel_coeff
        self.decay_mel_coeff  = getattr(train_args, "decay_mel_coeff", False)
        self.num_warmup_steps = getattr(train_args, "warmup_steps", 0)
        self._g_updates       = 0   # 只在真正更新 G 时 +1

    def configure_optimizers(self):
        disc_params = [
            {"params": self.disc.parameters()},
            {"params": self.multi_res_disc.parameters()},
        ]

        gen_params = [
            {"params": self.gen.parameters()},
        ]

        if self.optimizers == 'AdamW':
            self.gen_optimizer = torch.optim.AdamW(gen_params, lr=self.args.learning_rate, betas=(0.5, 0.999))
            self.disc_optimizer = torch.optim.AdamW(disc_params, lr=self.args.learning_rate, betas=(0.5, 0.999))
        else:
            self.gen_optimizer = torch.optim.Adam(gen_params, lr=self.args.learning_rate, betas=(0.5, 0.999))
            self.disc_optimizer = torch.optim.Adam(disc_params, lr=self.args.learning_rate, betas=(0.5, 0.999))

        if self.lr_scheduler_type == "linear":
            self.gen_scheduler = get_linear_schedule_with_warmup(
                self.gen_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.max_steps
            )
            self.disc_scheduler = get_linear_schedule_with_warmup(
                self.disc_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.max_steps
            )
        elif self.lr_scheduler_type == "cosine":
            self.gen_scheduler = get_cosine_schedule_with_warmup(
                self.gen_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.max_steps
            )
            self.disc_scheduler = get_cosine_schedule_with_warmup(
                self.disc_optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.max_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.args.lr_scheduler_type}")
        optimizers = [self.gen_optimizer, self.disc_optimizer]
        schedulers = [self.gen_scheduler, self.disc_scheduler]
        if self._ema is not None:
            schedulers.append(self._ema)
        
        self.optimizers = optimizers
        self.lr_schedulers = schedulers
    
    def optimizer_step(self, *args, **kwargs):
        "禁用 Trainer 的默认优化器步进逻辑"
        return

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = self.args.device
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    def _split_batch(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:

        mel  = inputs["mel_spectrogram"]
        meta = {k: inputs[k] for k in ("audio_path", "caption", "text_input_ids", "text_attention_mask") if k in inputs}
        return mel, meta

    # 是否噪声采样需要 cond（本例中 gen 不需要 noise；留个接口）
    def _needs_cond_in_noise(self) -> bool:
        if self.noise_sampler is None:
            return False
        return self.noise_sampler.__code__.co_argcount >= 2

    # ----------------------------------
    # 核心训练逻辑
    # ----------------------------------
    def training_step(self, 
                      model: nn.Module, 
                      inputs: Dict[str, Any], 
                      num_items: int | None = None
                      ) -> torch.Tensor:
        gen, disc, args = self.gen, self.disc, self.args
        inputs = self._prepare_inputs(inputs)
        mel, meta = self._split_batch(inputs)
        real = mel  # 若真实目标是 waveform，自行替换

        # 用于多分辨率判别器的可选参数
        bw_id = meta.get("bandwidth_id", None)

        if self._phase == "D":
            # -------- D step --------
            with torch.no_grad():
                fake, _, _ = gen(mel)

            # DiffAug 先做
            if self.use_diff_aug:
                real_aug = DiffAugment(real, policy="gain,shift,cutout,tmask", prob=0.5)
                fake_aug = DiffAugment(fake,  policy="gain,shift,cutout,tmask", prob=0.5)
            else:
                real_aug, fake_aug = real, fake

            self.disc_optimizer.zero_grad(set_to_none=True)

            # 主判别器
            print(f"real_aug shape: {real_aug.shape}")
            logits_real = disc(real_aug)
            print(f"logits_real shape: {logits_real.shape}")
            logits_fake = disc(fake_aug)
            print(f'fake_aug shape: {fake_aug.shape}')
            print(f"logits_fake shape: {logits_fake.shape}")
            # loss_d_main = self.disc_loss(logits_real, logits_fake, args.loss_type)
            loss_d_main, _, _ = self.disc_loss(logits_real, logits_fake)

            # 多分辨率判别器
            real_mrd, fake_mrd, _, _ = self.multi_res_disc(y_spec=real_aug, y_spec_hat=fake_aug.detach(), bandwidth_id=bw_id)
            loss_mrd, loss_mrd_real, _ = self.disc_loss(real_mrd, fake_mrd)
            loss_mrd = loss_mrd / len(loss_mrd_real)

            loss_d = loss_d_main + loss_mrd

            self.accelerator.backward(loss_d)
            self.disc_optimizer.step()
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()

            self.log({"loss_d": loss_d.detach(),
                    "loss_d_main": loss_d_main.detach(),
                    "loss_d_mrd": loss_mrd.detach()}
                    )

            if self.accelerator.sync_gradients:
                self._phase = "G"
            return loss_d.detach() * 0.0

        else:
            # -------- G step --------
            self.gen_optimizer.zero_grad(set_to_none=True)

            fake, codebook_loss, info = gen(mel)  # 不 detach
            if self.use_diff_aug:
                fake_aug = DiffAugment(fake, policy="color,translation,cutout", prob=0.5)
            else:
                fake_aug = fake

            # 主判别器对抗
            logits_fake = disc(fake_aug)
            loss_g_adv = self.gen_adv_loss(logits_fake) * getattr(args, "disc_weight", 1.0)

            # MRD 对抗 + FM
            _, fake_mrd, fmap_r_mrd, fmap_g_mrd = self.multi_res_disc(y_spec=real, y_spec_hat=fake, bandwidth_id=bw_id)
            loss_gen_mrd, list_mrd = self.gen_loss(disc_outputs=fake_mrd)
            loss_gen_mrd = loss_gen_mrd / len(list_mrd)

            loss_fm_mrd = self.feature_matching_loss(fmap_r=fmap_r_mrd, fmap_g=fmap_g_mrd) / len(fmap_r_mrd)
            if getattr(args, "fm_coeff", 0.0) > 0:
                loss_fm_mrd *= args.fm_coeff

            # Mel 重建
            loss_mel = torch.zeros((), device=real.device)
            if getattr(args, "mel_coeff", 0.0) > 0:
                loss_mel = self.rec_loss(fake, real) * self.mel_loss_coeff * self.rec_weight

            # VQ / commit / REPA
            loss_vq = torch.zeros((), device=real.device)
            if isinstance(codebook_loss, (list, tuple)) and len(codebook_loss) >= 3:
                loss_vq = (codebook_loss[0] + codebook_loss[1] + codebook_loss[2]) * getattr(args, "vq_coeff", 1.0)
                if len(codebook_loss) > 3:  # REPA
                    loss_vq += codebook_loss[3]

            loss_g = loss_g_adv + loss_gen_mrd + loss_fm_mrd + loss_mel + loss_vq

            self.accelerator.backward(loss_g)
            self.gen_optimizer.step()
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()
            if self._ema is not None:
                self._ema.update(gen)

            if self.accelerator.sync_gradients:          # 避免梯度累积期间重复更新
                self._g_updates += 1
                if self.decay_mel_coeff:
                    self.mel_loss_coeff = self.base_mel_coeff * self._mel_decay(self._g_updates)

            self.log({
                "loss_g": loss_g.detach(),
                "loss_g_adv": loss_g_adv.detach(),
                "loss_g_mrd": loss_gen_mrd.detach(),
                "loss_fm_mrd": loss_fm_mrd.detach(),
                "loss_mel": loss_mel.detach(),
                "loss_vq": loss_vq.detach(),
                "lr_g": self.gen_optimizer.param_groups[0]["lr"],
                "lr_d": self.disc_optimizer.param_groups[0]["lr"],
            })

            if self.accelerator.sync_gradients:
                self._phase = "D"

            return loss_g.detach()
    # ----------------------------------
    # Eval：用 EMA 权重生成对比
    # ----------------------------------
    def evaluation_loop(self, dataloader, description: str, prediction_loss_only: Optional[bool] = None):
        gen = self.gen
        gen.eval()

        ema_backup = None
        if self._ema is not None:
            ema_backup = {n: p.clone() for n, p in gen.state_dict().items()}
            self._ema.copy_to(gen)

        preds, labels = [], []
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            real, mel, _ = self._split_batch(inputs)
            with torch.no_grad():
                fake_dec, _, _ = self._gen_forward(mel)

            preds.append(fake_dec.cpu())
            labels.append(real.cpu())

        if self._ema is not None and ema_backup is not None:
            gen.load_state_dict(ema_backup, strict=True)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics((preds, labels))

        num_samples = len(dataloader.dataset) if has_length(dataloader.dataset) else len(labels)
        metrics = denumpify_detensorize(metrics)
        metrics["eval_samples"] = num_samples
        self.log(metrics)

        EvalOutput = type("EvalLoopOutput", (), {})
        out = EvalOutput()
        out.metrics = metrics
        out.predictions = None
        out.label_ids = None
        return out

    # ----------------------------------
    # Save / Load
    # ----------------------------------
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        output_dir = output_dir or self.args.output_dir
        torch.save(self.disc.state_dict(), os.path.join(output_dir, "discriminator.pt"))
        if self._ema is not None:
            torch.save(self._ema.shadow, os.path.join(output_dir, "ema_shadow.pt"))

    def _load_from_checkpoint(self, resume_from_checkpoint):
        # 只在你自定义恢复时用；通常 HF 的 load_from_checkpoint 会调用 create_optimizer_and_scheduler
        disc_pth = os.path.join(resume_from_checkpoint, "discriminator.pt")
        if os.path.exists(disc_pth):
            self.disc.load_state_dict(torch.load(disc_pth, map_location="cpu"))
        ema_pth = os.path.join(resume_from_checkpoint, "ema_shadow.pt")
        if self._ema is not None and os.path.exists(ema_pth):
            self._ema.shadow = torch.load(ema_pth, map_location="cpu")

    def _mel_decay(self, step: int, num_cycles: float = 0.5) -> float:
        # 你 Lightning 里用 max_steps//2，因为两优化器各走一次；
        # 这里真正的 G 步数 = self.max_steps//2（或根据你自己逻辑）
        total_g_steps = (self.max_steps or self.state.max_steps) // 2
        if step < self.num_warmup_steps:
            return 1.0
        progress = (step - self.num_warmup_steps) / max(1, total_g_steps - self.num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))


from torch.utils.data import Dataset

# ---------- 轻量随机数据集 ----------
class RandomMelDataset(Dataset):
    def __init__(self, num_samples=32, n_mels=128, t=1000):
        self.data = torch.randn(num_samples, 1, n_mels, t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "mel_spectrogram": self.data[idx],
            "audio_path": f"{idx}.wav",
        }

# ---------- GANTrainingArguments ----------
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class GANTrainingArguments(TrainingArguments):
    # -------- 额外 GAN 超参数 --------
    ema: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    disc_weight: float = field(default=1.0)
    fm_coeff: float = field(default=0.0)
    mel_coeff: float = field(default=45.0)
    decay_mel_coeff: bool = field(default=False)
    vq_coeff: float = field(default=1.0)
    warmup_steps: int = field(default=0)

    # ★ 关键：保留父类的后置初始化 ★
    def __post_init__(self):
        super().__post_init__()      # 让 distributed_state、device 等被正确设置
        # 你自己的额外校验 / 日志（可选）


# ---------- main ----------
def main():
    # 1. 实例化模型
    config = SoftVQConfig.from_pretrained("configs/softvq_config.json")
    gen = SoftVQModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen.to(device)

    # disc = DummyDiscriminator()
    from models.discriminators import DinoDiscriminator
    disc = DinoDiscriminator()
    disc.to(device)

    # 2. 训练参数
    args = GANTrainingArguments(
        output_dir="debug_runs",
        per_device_train_batch_size=4,
        num_train_epochs=1,          # 只跑 1 epoch
        learning_rate=2e-4,
        logging_steps=1,
        evaluation_strategy="no",    # 先关闭 eval
        report_to=[],                # 禁掉 wandb 等
        remove_unused_columns=False,
        save_strategy="no",
        max_steps=10,                # 跑 10 step 演示
        # device=device,
    )

    # 3. 数据
    train_ds = RandomMelDataset()

    # 4. Trainer（关键：添加 train_dataset=train_ds）
    trainer = SoftVQ_GANTrainer(
        generator=gen,
        discriminator=disc,
        noise_sampler=None,
        train_args=args,
        max_steps=args.max_steps,
        train_dataset=train_ds,   # ← 必需！
    )
    trainer.configure_optimizers()  # 初始化优化器和调度器

    # 5. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
