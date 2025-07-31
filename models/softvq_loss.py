import torch
import torch.nn.functional as F
import torch.distributed as tdist
import torchaudio
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from .discriminators import PatchGANDiscriminator, StyleGANDiscriminator, PatchGANMaskBitDiscriminator, DinoDiscriminator 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from typing import List, Tuple

from dataclasses import dataclass

def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


class ReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """
    def __init__(self, loss_type: str = "l1"):
        """
        Args:
            loss_type (str): Type of loss to use. Currently only "l1" is supported.
        """
        super().__init__()
        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "l2":
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Supported types are 'l1' and 'l2'.")

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: LLoss between the mel-scaled magnitude spectrograms.
        """

        loss = self.loss_fn(y_hat, y)
        return loss


class GeneratorLoss(nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)

# class GAN_Adv_Loss(nn.Module):
#     """
#     Computes a discriminator loss, given a discriminator on
#     generated waveforms/spectrograms compared to ground truth
#     waveforms/spectrograms. Computes the loss for both the
#     discriminator and the generator in separate functions.
#     """

#     def __init__(self, discriminator):
#         super().__init__()
#         self.discriminator = discriminator

#     def forward(self, fake, real):
#         # d_fake = self.discriminator(fake.audio_data)
#         # d_real = self.discriminator(real.audio_data)
#         d_fake = self.discriminator(fake)
#         d_real = self.discriminator(real)
#         return d_fake, d_real

#     def discriminator_loss(self, fake, real):
#         d_fake, d_real = self.forward(fake.clone().detach(), real)

#         loss_d = 0
#         for x_fake, x_real in zip(d_fake, d_real):
#             loss_d += torch.mean(x_fake[-1] ** 2)
#             loss_d += torch.mean((1 - x_real[-1]) ** 2)
#         return loss_d

#     def generator_loss(self, fake, real):
#         d_fake, d_real = self.forward(fake, real)

#         loss_g = 0
#         for x_fake in d_fake:
#             loss_g += torch.mean((1 - x_fake[-1]) ** 2)

#         loss_feature = 0

#         for i in range(len(d_fake)):
#             for j in range(len(d_fake[i]) - 1):
#                 loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
#         return loss_g, loss_feature


# ---------------------------
# 1. 配置
# ---------------------------
@dataclass
class Wav2vecCfg:
    encoder_embed_dim: int = 768
    final_dim: int = -1                 # <=0 时使用 encoder_embed_dim
    num_negatives: int = 100
    cross_sample_negatives: int = 0
    logit_temp: float = 0.1


# ---------------------------
# 2. 纯 torch 的 Criterion
# ---------------------------
class Wav2vecCriterionTorch(nn.Module):
    """
    InfoNCE-style contrastive loss used in wav2vec/Hubert.

    Args:
        cfg: Wav2vecCfg
        infonce: keep for compatibility (always True here)
        reduction: "sum" or "mean"
    """
    def __init__(self, cfg: Wav2vecCfg, infonce: bool = True, reduction: str = "sum"):
        super().__init__()
        self.cfg = cfg
        self.infonce = infonce
        self.reduction = reduction

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        self.project_y = nn.Linear(cfg.encoder_embed_dim, final_dim)
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # cache for masked_fill (无需 XLA 特判)
        self._neg_inf = float("-inf")

    @torch.no_grad()
    def _arange_like(self, length: int, device: torch.device):
        return torch.arange(length, device=device)

    def forward(self, cnn_feat: torch.Tensor,
                mask_indices: torch.Tensor,
                quantized: torch.Tensor,
                reduce: bool = True) -> torch.Tensor:
        """
        cnn_feat  : [B, T, D]
        mask_indices : [B, T] bool
        quantized : [B, T, D] or [B, D, T] (这里假设输入和原版一样: [B, T, D], 若不是会转置)
        """
        if quantized.dim() == 3 and quantized.shape[1] != cnn_feat.shape[1]:
            # 原代码 quantized.transpose(1,2)，说明原始是 [B, D, T]
            quantized = quantized.transpose(1, 2)

        B, T, D = cnn_feat.shape
        # 1) 取目标 y (unmasked cnn feats at masked positions)
        y = cnn_feat[mask_indices].view(B, -1, D)  # [B, M, D]
        y = self.project_y(y)

        # 2) 采样负样本
        negs, _ = self.sample_negatives(y, num=y.size(1), padding_count=None)  # [N_neg, B, M, D]

        # 3) 取预测 x (reconstructed tokens)
        x = quantized[mask_indices].view(B, -1, D)  # [B, M, D]
        x = self.final_proj(x)

        # 4) 计算 logits_3d : [1+N_neg, B, M]
        logits_3d = self.compute_preds(x, y, negs)

        # 5) reshape -> logits [B*M, 1+N_neg], targets [B*M]
        logits = self.get_logits(logits_3d).float()
        targets = self.get_targets(logits_3d)

        loss = F.cross_entropy(logits, targets, reduction="sum" if reduce else "none")
        if self.reduction == "mean" and reduce:
            loss = loss / targets.numel()

        return loss

    def compute_preds(self, x: torch.Tensor, y: torch.Tensor, negatives: torch.Tensor):
        """
        x: [B, M, D]  (masked quantized codes)
        y: [B, M, D]  (unmasked cnn feat at those masked positions)
        negatives: [N_neg, B, M, D]
        return: logits [1+N_neg, B, M]
        """
        # 判断负样本中有没有与正样本完全相同的（防止采到正样本）
        neg_is_pos = (y.unsqueeze(0) == negatives).all(-1)  # [N_neg, B, M]

        # 拼接 targets
        targets = torch.cat([y.unsqueeze(0), negatives], dim=0)  # [1+N_neg, B, M, D]

        # 归一化后做余弦相似度（也可直接 cosine_similarity）
        x_norm = F.normalize(x.float(), dim=-1)
        t_norm = F.normalize(targets.float(), dim=-1)
        # 广播： (1+N_neg, B, M, D) 与 (B, M, D) -> 手动展开
        # 用逐维余弦相似度更直观，也可:
        # logits = (t_norm * x_norm.unsqueeze(0)).sum(-1)
        logits = torch.cosine_similarity(x_norm.unsqueeze(0), t_norm, dim=-1)  # [1+N_neg, B, M]
        logits = logits / self.cfg.logit_temp
        logits = logits.to(dtype=x.dtype)

        # 把误采样为正样本的位置置为 -inf
        if neg_is_pos.any():
            logits[1:].masked_fill_(neg_is_pos, self._neg_inf)

        return logits

    @staticmethod
    def get_logits(x: torch.Tensor):
        # x: [1+N_neg, B, M] -> [B*M, 1+N_neg]
        x = x.transpose(0, 2)  # [M, B, 1+N_neg]
        x = x.reshape(-1, x.size(-1))
        return x

    @staticmethod
    def get_targets(x: torch.Tensor):
        # 正样本都是 index=0
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    @torch.no_grad()
    def sample_negatives(self, y: torch.Tensor, num: int, padding_count=None):
        """
        y: [B, M, D]
        num: M (masked steps per sample)
        Returns:
            negs: [N_neg, B, M, D]
            neg_idxs: [B, num_neg_total * M]
        """
        if self.cfg.num_negatives == 0 and self.cfg.cross_sample_negatives == 0:
            return y.new_zeros((0,)), None

        B, M, D = y.shape
        y_flat = y.view(-1, D)      # [B*M, D]

        cross_high = M * B
        high = M - (padding_count or 0)
        assert high > 1, f"{B,M,D}"

        device = y.device
        # 1) from same sample
        neg_idxs = None
        if self.cfg.num_negatives > 0:
            tszs = self._arange_like(num, device).unsqueeze(-1).expand(-1, self.cfg.num_negatives).flatten()
            # [M * num_negatives]
            neg_idxs_same = torch.randint(low=0, high=high - 1, size=(B, self.cfg.num_negatives * num), device=device)
            # 避免 index == 正样本位置
            neg_idxs_same[neg_idxs_same >= tszs] += 1
            # 偏移到每个 batch 的段
            neg_idxs_same = neg_idxs_same + (torch.arange(B, device=device).unsqueeze(1) * high)
            neg_idxs = neg_idxs_same

        # 2) from other samples
        if self.cfg.cross_sample_negatives > 0:
            tszs = self._arange_like(num, device).unsqueeze(-1).expand(-1, self.cfg.cross_sample_negatives).flatten()
            cross_neg_idxs = torch.randint(low=0, high=cross_high - 1,
                                           size=(B, self.cfg.cross_sample_negatives * num),
                                           device=device)
            cross_neg_idxs[cross_neg_idxs >= tszs] += 1
            neg_idxs = cross_neg_idxs if neg_idxs is None else torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        # gather
        negs = y_flat[neg_idxs.view(-1)]  # [B * num_total * M, D]
        total_negs = self.cfg.num_negatives + self.cfg.cross_sample_negatives
        negs = negs.view(B, num, total_negs, D).permute(2, 0, 1, 3)  # [N_neg, B, M, D]

        return negs, neg_idxs
