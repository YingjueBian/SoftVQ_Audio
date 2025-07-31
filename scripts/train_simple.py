#!/usr/bin/env python3
"""
Simplified training script for AIMv2 multimodal model.
Training only - no validation or inference.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import set_seed, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from aimv2.model.multimodal_model import AIMv2MultimodalModel, AIMv2MultimodalConfig
from aimv2.datasets.simple_laion_dataset import SimpleLaionDataset, SimpleCollator


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        
        return True, rank, world_size, gpu
    else:
        return False, 0, 1, 0


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    device,
    accumulation_steps=1,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                audio_patches=batch["audio_patches"],
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_masks"],
                return_loss=True,
            )
            loss = outputs["loss"] / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Track loss
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log to wandb
        if batch_idx % 100 == 0 and wandb.run is not None:
            wandb.log({
                "train/loss": loss.item() * accumulation_steps,
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": epoch * len(dataloader) + batch_idx,
            })
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--config_path", type=str, default="aimv2/configs/multimodal_config.json")
    parser.add_argument("--output_dir", type=str, default="aimv2/checkpoints/multimodal")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, 
                       default="/zoomai/colddata/asr/dean/sound_dataset/laion-audio-300m-arrow/train")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5000)
    
    # Other arguments
    parser.add_argument("--wandb_project", type=str, default="aimv2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup distributed
    is_distributed, rank, world_size, gpu = setup_distributed()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb on main process
    if rank == 0 and args.wandb_project:
        wandb.init(project=args.wandb_project, config=args)
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    print("Creating model...")
    model_config = AIMv2MultimodalConfig(**config["model"])
    model = AIMv2MultimodalModel(model_config)
    model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[gpu])
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = SimpleLaionDataset(
        dataset_path=args.dataset_path,
        **config.get("dataset", {})
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=SimpleCollator(),
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    
    total_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            scaler,
            epoch,
            device,
            args.gradient_accumulation_steps,
        )
        
        print(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if rank == 0:
            checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model
            if is_distributed:
                model.module.save_pretrained(checkpoint_path)
            else:
                model.save_pretrained(checkpoint_path)
            
            # Save training state
            torch.save({
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, os.path.join(checkpoint_path, "training_state.pt"))
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    if rank == 0:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        
        if is_distributed:
            model.module.save_pretrained(final_path)
        else:
            model.save_pretrained(final_path)
            
        print(f"Training complete! Final model saved to {final_path}")
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()