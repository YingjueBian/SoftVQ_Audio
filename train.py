#!/usr/bin/env python3
"""
Training script for AIMv2 Speech model on LAION-Audio dataset.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from models import AIMv2Config, AIMv2Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIMv2DataCollator:
    """Data collator for AIMv2 training"""
    
    def __init__(self, tokenizer, max_text_length=77):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack mel spectrograms
        mel_spectrograms = torch.stack([f["mel_spectrogram"] for f in features])
        
        # Process text
        texts = [f["caption"] for f in features]
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        
        return {
            "mel_spectrogram": mel_spectrograms,
            "input_ids": text_inputs["input_ids"],
        }


class AIMv2Trainer(Trainer):
    """Custom trainer for AIMv2"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Log losses
        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {}
            if outputs.speech_loss is not None:
                metrics["speech_loss"] = outputs.speech_loss.item()
            if outputs.text_loss is not None:
                metrics["text_loss"] = outputs.text_loss.item()
            if loss is not None:
                metrics["total_loss"] = loss.item()
            self.log(metrics)
        
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/aimv2_speech_config.json")
    parser.add_argument("--output_dir", type=str, default="outputs/aimv2_speech")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    args = parser.parse_args()
    
    # Initialize distributed training if needed
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Initialized process group, local_rank: {args.local_rank}")
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Set seed
    set_seed(42)
    
    # Create model
    logger.info("Creating AIMv2 model...")
    model_config = AIMv2Config(
        encoder_embed_dim=config_dict["encoder_embed_dim"],
        encoder_depth=config_dict["encoder_depth"],
        encoder_num_heads=config_dict["encoder_num_heads"],
        encoder_mlp_ratio=config_dict["encoder_mlp_ratio"],
        decoder_dim=config_dict["decoder_dim"],
        decoder_depth=config_dict["decoder_depth"],
        decoder_num_heads=config_dict["decoder_num_heads"],
        decoder_ffn_dim=config_dict["decoder_ffn_dim"],
        n_mels=config_dict["n_mels"],
        max_time_frames=config_dict["max_time_frames"],
        patch_size=tuple(config_dict["patch_size"]),
        speech_patch_dim=config_dict["speech_patch_dim"],
        vocab_size=config_dict["vocab_size"],
        max_seq_length=config_dict["max_seq_length"],
        speech_loss_weight=config_dict["speech_loss_weight"],
        text_loss_weight=config_dict["text_loss_weight"],
        dropout=config_dict["dropout"],
        norm_eps=config_dict["norm_eps"],
    )
    model = AIMv2Model(model_config)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dict["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data collator with dynamic padding
    from data.laion_audio import LaionAudioCollator
    collator = LaionAudioCollator(pad_token_id=tokenizer.pad_token_id)
    
    # Create datasets
    logger.info(f"Loading dataset from {config_dict['dataset_path']}")
    
    # For distributed training, we'll let the Trainer handle the DataLoader creation
    # So we just need to create the dataset
    from data.laion_audio import LaionAudioDataset
    
    train_dataset = LaionAudioDataset(
        dataset_path=config_dict["dataset_path"],
        split=config_dict["train_split"],
        sample_rate=config_dict["sample_rate"],
        n_mels=config_dict["n_mels"],
        hop_length=config_dict["hop_length"],
        max_duration_seconds=config_dict["max_audio_length"],
        tokenizer_name=config_dict["tokenizer_name"],
        max_text_length=config_dict["max_text_length"],
        augment=True,
    )
    
    eval_dataset = None
    eval_path = os.path.join(config_dict["dataset_path"], config_dict["eval_split"])
    if os.path.exists(eval_path):
        eval_dataset = LaionAudioDataset(
            dataset_path=config_dict["dataset_path"],
            split=config_dict["eval_split"],
            sample_rate=config_dict["sample_rate"],
            n_mels=config_dict["n_mels"],
            hop_length=config_dict["hop_length"],
            max_duration_seconds=config_dict["max_audio_length"],
            tokenizer_name=config_dict["tokenizer_name"],
            max_text_length=config_dict["max_text_length"],
            augment=False,
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config_dict["num_train_epochs"],
        per_device_train_batch_size=config_dict["batch_size"],
        per_device_eval_batch_size=config_dict["batch_size"],
        gradient_accumulation_steps=config_dict["gradient_accumulation_steps"],
        learning_rate=config_dict["learning_rate"],
        warmup_steps=config_dict["warmup_steps"],
        weight_decay=config_dict["weight_decay"],
        adam_beta1=config_dict["adam_beta1"],
        adam_beta2=config_dict["adam_beta2"],
        max_grad_norm=config_dict["max_grad_norm"],
        logging_steps=config_dict["logging_steps"],
        save_steps=config_dict["save_steps"],
        eval_steps=config_dict["eval_steps"] if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        fp16=config_dict["fp16"],
        gradient_checkpointing=config_dict["gradient_checkpointing"],
        dataloader_num_workers=config_dict["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if args.deepspeed else None,
    )
    
    # Create trainer
    trainer = AIMv2Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()