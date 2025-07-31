import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk, Dataset as HFDataset
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import os
from pathlib import Path
import torchaudio
from transformers import AutoTokenizer

from process.audio_features import (
    AudioFeatureExtractor,
    AudioPatchExtractor,
    pad_or_truncate_audio,
    load_audio,
)
from process.augmentation import AudioAugmentation, SpecAugment, RandAugment


class LaionAudioDataset(Dataset):
    """
    Optimized dataset loader for LAION-Audio-300M using HuggingFace datasets.
    Supports save_to_disk functionality and efficient loading.
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        hf_dataset_name: str = "laion/LAION-Audio-300M",
        split: str = "train",
        cache_dir: Optional[str] = None,
        max_duration_seconds: float = 12.0,
        sample_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 512,
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        tokenizer_name: str = "bert-base-uncased",
        max_text_length: int = 77,
        augment: bool = True,
        augment_prob: float = 0.5,
        spec_augment: bool = True,
        rand_augment: bool = False,
        rand_augment_n: int = 2,
        rand_augment_m: int = 10,
        streaming: bool = False,
    ):
        self.dataset_path = dataset_path
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_audio_length = int(max_duration_seconds * sample_rate)
        self.augment = augment
        self.streaming = streaming
        
        # Initialize feature extractors
        self.audio_feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        )
        
        self.patch_extractor = AudioPatchExtractor(
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        
        # Initialize tokenizer for text
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_length = max_text_length
        
        # Initialize augmentation
        if augment:
            self.audio_augmentation = AudioAugmentation(
                sample_rate=sample_rate,
                augment_prob=augment_prob,
            )
            
            if spec_augment:
                self.spec_augment = SpecAugment()
            else:
                self.spec_augment = None
                
            if rand_augment:
                self.rand_augment = RandAugment(
                    n=rand_augment_n,
                    m=rand_augment_m,
                    sample_rate=sample_rate,
                )
            else:
                self.rand_augment = None
        else:
            self.audio_augmentation = None
            self.spec_augment = None
            self.rand_augment = None
            
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset from disk or HuggingFace hub."""
        if self.dataset_path and os.path.exists(self.dataset_path):
            # Load from local disk
            print(f"Loading dataset from disk: {self.dataset_path}")
            self.dataset = load_from_disk(self.dataset_path)
            if isinstance(self.dataset, dict) and self.split in self.dataset:
                self.dataset = self.dataset[self.split]
        else:
            # Load from HuggingFace hub
            print(f"Loading dataset from HuggingFace: {self.hf_dataset_name}")
            self.dataset = load_dataset(
                self.hf_dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
            )
            
    def save_to_disk(self, save_path: str):
        """Save the dataset to disk for faster loading."""
        if not self.streaming:
            print(f"Saving dataset to disk: {save_path}")
            self.dataset.save_to_disk(save_path)
        else:
            print("Cannot save streaming dataset to disk.")
            
    def _load_audio_file(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess audio file."""
        try:
            # Load audio
            waveform, orig_sr = load_audio(
                audio_path,
                sample_rate=self.sample_rate,
                mono=True,
                normalize=True,
            )
            
            # Pad or truncate to max length
            waveform = pad_or_truncate_audio(waveform, self.max_audio_length)
            
            return waveform
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None
            
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and process text caption."""
        # Tokenize text without padding (will be handled in collator)
        encoding = self.tokenizer(
            text,
            padding=False,  # No padding here
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
    def __len__(self) -> int:
        if self.streaming:
            # For streaming datasets, we can't know the exact length
            return float('inf')
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        sample = self.dataset[idx]
        
        # Handle LAION-Audio-300M format
        # Check if audio is provided as array (LAION-Audio-300M format)
        if "audio.mp3" in sample and isinstance(sample["audio.mp3"], dict) and "array" in sample["audio.mp3"]:
            # Audio is pre-loaded as array
            audio_data = sample["audio.mp3"]
            waveform = torch.from_numpy(audio_data["array"]).float().detach()
            orig_sr = audio_data.get("sampling_rate", 16000)
            
            # Resample if necessary
            if orig_sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_sr, self.sample_rate)
            
            # Get caption from metadata
            metadata = sample.get("metadata.json", {})
            caption = metadata.get("caption", "")
            audio_path = audio_data.get("path", f"sample_{idx}")
        else:
            # Fallback to old format
            audio_path = sample.get("audio", {}).get("path") or sample.get("audio_path")
            caption = sample.get("text") or sample.get("caption") or ""
            
            # Load audio from file
            waveform = self._load_audio_file(audio_path)
            if waveform is None:
                # Return a dummy sample if audio loading fails
                waveform = torch.zeros(self.max_audio_length)
        
        # Ensure waveform is 1D
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        
        # Pad or truncate to max length
        waveform = pad_or_truncate_audio(waveform, self.max_audio_length)
            
        # Apply audio augmentation if enabled
        if self.augment and self.audio_augmentation:
            if self.rand_augment:
                waveform = self.rand_augment(waveform)
            else:
                waveform = self.audio_augmentation(waveform)
                
        # Extract mel-spectrogram features
        mel_spec = self.audio_feature_extractor(waveform)
        
        # Apply spectrogram augmentation if enabled
        if self.augment and self.spec_augment:
            mel_spec = self.spec_augment(mel_spec)
            
        # Ensure mel_spec has shape (n_mels, time_frames)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)  # Remove batch dimension
            
        # Pad or truncate mel-spectrogram to fixed time dimension
        # Use a fixed number of time frames to ensure consistent patch count
        # With n_mels=128 and patch_size=(16,16), we need time_frames to be divisible by 16
        expected_time_frames = 1008  # This gives us exactly 63 time patches (1008/16=63)
        current_time_frames = mel_spec.shape[-1]
        
        if current_time_frames < expected_time_frames:
            # Pad with zeros
            padding = expected_time_frames - current_time_frames
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding), value=0)
        elif current_time_frames > expected_time_frames:
            # Truncate
            mel_spec = mel_spec[..., :expected_time_frames]
            
        # Extract patches
        patches = self.patch_extractor(mel_spec.unsqueeze(0))  # Add batch dim for patch extractor
        
        # Process text
        text_data = self._process_text(caption)
        
        # Add channel dimension to get (C, H, W) format expected by model
        # where C=1 for single channel audio
        mel_spec = mel_spec.unsqueeze(0)
        
        return {
            "audio_patches": (patches.squeeze(0) if patches.dim() > 2 else patches).detach(),
            "mel_spectrogram": mel_spec.detach(),  # Shape: (1, n_mels, time_frames)
            "input_ids": text_data["input_ids"].detach(),
            "text_attention_mask": text_data["attention_mask"].detach(),
            "audio_path": audio_path,
            "caption": caption,
        }


class LaionAudioCollator:
    """Custom collator for batching LAION-Audio samples with dynamic padding."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples with dynamic padding for text."""
        # Stack mel spectrograms (fixed size)
        mel_spectrograms = torch.stack([sample["mel_spectrogram"] for sample in batch])
        
        # Dynamic padding for text inputs
        input_ids_list = [sample["input_ids"] for sample in batch]
        attention_mask_list = [sample["text_attention_mask"] for sample in batch]
        
        # Find max length in batch
        max_length = max(ids.size(0) for ids in input_ids_list)
        
        # Pad sequences to max length in batch
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids_list, attention_mask_list):
            padding_length = max_length - ids.size(0)
            if padding_length > 0:
                # Pad input_ids with pad_token_id
                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=ids.dtype)
                ])
                # Pad attention_mask with zeros
                padded_mask = torch.cat([
                    mask,
                    torch.zeros(padding_length, dtype=mask.dtype)
                ])
            else:
                padded_ids = ids
                padded_mask = mask
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        # Stack padded sequences
        input_ids = torch.stack(padded_input_ids)
        attention_masks = torch.stack(padded_attention_masks)
        
        # Collect metadata (optional, for debugging)
        captions = [sample["caption"] for sample in batch]
        
        return {
            "mel_spectrogram": mel_spectrograms,
            "input_ids": input_ids,
            "attention_mask": attention_masks,  # Include attention mask for proper handling
            "caption": captions,  # Keep for debugging
        }


def create_laion_audio_dataloader(
    dataset_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for LAION-Audio dataset.
    
    Args:
        dataset_path: Path to saved dataset or None to load from HuggingFace
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the dataset
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop the last incomplete batch
        **dataset_kwargs: Additional arguments for LaionAudioDataset
        
    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = LaionAudioDataset(
        dataset_path=dataset_path,
        **dataset_kwargs,
    )
    
    # Create collator
    collator = LaionAudioCollator()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )
    
    return dataloader


def prepare_laion_audio_splits(
    dataset_path: str,
    save_dir: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
):
    """
    Prepare train/val/test splits for LAION-Audio dataset.
    
    Args:
        dataset_path: Path to the full dataset
        save_dir: Directory to save the splits
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
    """
    # Load full dataset
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset("laion/LAION-Audio-300M", cache_dir=dataset_path)
        
    # Create splits
    train_test_split = dataset.train_test_split(
        test_size=(val_ratio + test_ratio),
        seed=seed,
    )
    
    train_dataset = train_test_split["train"]
    temp_dataset = train_test_split["test"]
    
    # Split temp into val and test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_test_split = temp_dataset.train_test_split(
        test_size=val_test_ratio,
        seed=seed,
    )
    
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]
    
    # Save splits
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving train split ({len(train_dataset)} samples)...")
    train_dataset.save_to_disk(os.path.join(save_dir, "train"))
    
    print(f"Saving validation split ({len(val_dataset)} samples)...")
    val_dataset.save_to_disk(os.path.join(save_dir, "validation"))
    
    print(f"Saving test split ({len(test_dataset)} samples)...")
    test_dataset.save_to_disk(os.path.join(save_dir, "test"))
    
    print("Dataset splits saved successfully!")