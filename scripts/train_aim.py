#!/usr/bin/env python3
"""
Training script for AIMv2 models with multi-GPU support.
Supports both audio classification and multimodal training.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
)
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from aimv2.model.aim_model import AIMv2Config, AIMv2ForAudioClassification
from aimv2.model.multimodal_model import AIMv2MultimodalModel, AIMv2MultimodalConfig
from aimv2.datasets.laion_audio import (
    create_laion_audio_dataloader,
    prepare_laion_audio_splits,
)
from aimv2.trainer.aim_trainer import (
    AIMv2Trainer,
    MultimodalAIMv2Trainer,
    compute_audio_classification_metrics,
    compute_retrieval_metrics,
    create_training_arguments,
)


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_JOBID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return True, rank, world_size, gpu


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_model(model_type: str, config: Dict[str, Any]):
    """Create model based on type and configuration."""
    if model_type == "classification":
        model_config = AIMv2Config(**config["model"])
        model_config.num_labels = config.get("num_labels", 527)  # AudioSet classes
        model = AIMv2ForAudioClassification(model_config)
    elif model_type == "multimodal":
        model_config = AIMv2MultimodalConfig(**config["model"])
        model = AIMv2MultimodalModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model, model_config


def main():
    parser = argparse.ArgumentParser(description="Train AIMv2 models")
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="multimodal",
        choices=["classification", "multimodal"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/zoomai/colddata/asr/dean/sound_dataset/laion-audio-300m-arrow",
        help="Path to LAION-Audio dataset",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        help="Name of evaluation split",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config file",
    )
    
    # Other arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="aimv2",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--prepare_splits",
        action="store_true",
        help="Prepare train/val/test splits from full dataset",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="standard",
        choices=["standard", "webdataset"],
        help="Dataset format type",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup distributed training
    is_distributed, rank, world_size, gpu = setup_distributed()
    is_main_process = rank == 0
    
    # Initialize wandb on main process
    if is_main_process and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{args.model_type}_{args.output_dir.split('/')[-1]}",
            config=args,
        )
    
    # Prepare splits if requested
    if args.prepare_splits and is_main_process:
        print("Preparing dataset splits...")
        prepare_laion_audio_splits(
            dataset_path=args.dataset_path,
            save_dir=args.dataset_path,
            train_ratio=0.9,
            val_ratio=0.05,
            test_ratio=0.05,
            seed=args.seed,
        )
        
    # Wait for main process to finish preparing splits
    if is_distributed:
        dist.barrier()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model, model_config = create_model(args.model_type, config)
    
    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda(gpu if is_distributed else 0)
        
    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[gpu])
    
    # Create datasets
    print("Creating datasets...")
    dataset_kwargs = config.get("dataset", {})
    
    # Choose dataloader based on dataset format
    if args.dataset_format == "webdataset":
        from aimv2.datasets.laion_audio_webdataset import create_webdataset_dataloader
        create_dataloader_fn = create_webdataset_dataloader
    else:
        create_dataloader_fn = create_laion_audio_dataloader
    
    train_dataloader = create_dataloader_fn(
        dataset_path=args.dataset_path,
        split=args.train_split,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        **dataset_kwargs,
    )
    
    eval_dataloader = create_dataloader_fn(
        dataset_path=args.dataset_path,
        split=args.eval_split,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        **dataset_kwargs,
    )
    
    # Setup training arguments
    training_args = create_training_arguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss" if args.model_type == "classification" else "eval_r@5",
        greater_is_better=args.model_type == "multimodal",
    )
    
    # Select appropriate trainer and metrics
    if args.model_type == "classification":
        trainer_class = AIMv2Trainer
        compute_metrics = compute_audio_classification_metrics
    else:
        trainer_class = MultimodalAIMv2Trainer
        compute_metrics = compute_retrieval_metrics
        
    # Create trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        compute_metrics=compute_metrics,
        use_wandb=is_main_process and args.wandb_project is not None,
    )
    
    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    if is_main_process:
        print(f"Saving final model to {args.output_dir}")
        trainer.save_model()
        
        # Save model config
        model_config.save_pretrained(args.output_dir)
        
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()
        
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()