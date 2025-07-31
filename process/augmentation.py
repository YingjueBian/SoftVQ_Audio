import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional, Tuple, List, Union
import random


class AudioAugmentation:
    """
    Audio augmentation techniques for AIMv2 training.
    Based on best practices from audio classification literature.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        augment_prob: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.augment_prob = augment_prob
        
    def time_stretch(
        self,
        waveform: torch.Tensor,
        rate: Optional[float] = None,
        min_rate: float = 0.8,
        max_rate: float = 1.2,
    ) -> torch.Tensor:
        """Apply time stretching to waveform."""
        if rate is None:
            rate = random.uniform(min_rate, max_rate)
            
        if rate == 1.0:
            return waveform
            
        # Use phase vocoder for time stretching
        stretch = T.TimeStretch(
            hop_length=512,
            n_freq=1025,  # (n_fft // 2) + 1
            fixed_rate=rate,
        )
        
        # Convert to spectrogram, apply stretch, convert back
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=waveform.device)
        
        spec = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        stretched_spec = stretch(spec)
        stretched_waveform = torch.istft(
            stretched_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
        )
        
        return stretched_waveform.detach()
    
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        n_steps: Optional[int] = None,
        min_steps: int = -4,
        max_steps: int = 4,
    ) -> torch.Tensor:
        """Apply pitch shifting to waveform."""
        if n_steps is None:
            n_steps = random.randint(min_steps, max_steps)
            
        if n_steps == 0:
            return waveform
            
        pitch_shift = T.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=n_steps,
        )
        
        return pitch_shift(waveform)
    
    def add_noise(
        self,
        waveform: torch.Tensor,
        noise_level: Optional[float] = None,
        min_noise: float = 0.001,
        max_noise: float = 0.01,
    ) -> torch.Tensor:
        """Add Gaussian noise to waveform."""
        if noise_level is None:
            noise_level = random.uniform(min_noise, max_noise)
            
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def volume_change(
        self,
        waveform: torch.Tensor,
        gain_db: Optional[float] = None,
        min_gain_db: float = -20,
        max_gain_db: float = 10,
    ) -> torch.Tensor:
        """Apply volume change to waveform."""
        if gain_db is None:
            gain_db = random.uniform(min_gain_db, max_gain_db)
            
        gain = T.Vol(gain_db, gain_type="db")
        return gain(waveform)
    
    def time_masking(
        self,
        waveform: torch.Tensor,
        mask_param: int = 50,
        num_masks: int = 1,
    ) -> torch.Tensor:
        """Apply time masking to waveform."""
        length = waveform.shape[-1]
        
        for _ in range(num_masks):
            mask_length = random.randint(1, min(mask_param, length // 10))
            mask_start = random.randint(0, length - mask_length)
            waveform[..., mask_start:mask_start + mask_length] = 0
            
        return waveform
    
    def mixup(
        self,
        waveform1: torch.Tensor,
        waveform2: torch.Tensor,
        alpha: float = 0.2,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply mixup augmentation between two waveforms.
        
        Returns:
            Tuple of (mixed_waveform, lambda_value)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        # Ensure both waveforms have the same length
        min_length = min(waveform1.shape[-1], waveform2.shape[-1])
        waveform1 = waveform1[..., :min_length]
        waveform2 = waveform2[..., :min_length]
        
        mixed = lam * waveform1 + (1 - lam) * waveform2
        
        return mixed, lam
    
    def __call__(
        self,
        waveform: torch.Tensor,
        augmentations: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Apply augmentations to waveform.
        
        Args:
            waveform: Audio waveform tensor
            augmentations: List of augmentation names to apply.
                          If None, randomly select from all available.
                          
        Returns:
            Augmented waveform
        """
        if random.random() > self.augment_prob:
            return waveform
            
        available_augmentations = [
            "time_stretch",
            "pitch_shift",
            "add_noise",
            "volume_change",
            "time_masking",
        ]
        
        if augmentations is None:
            # Randomly select augmentations
            num_augmentations = random.randint(1, 3)
            augmentations = random.sample(available_augmentations, num_augmentations)
            
        # Apply augmentations in sequence
        for aug_name in augmentations:
            if aug_name == "time_stretch":
                waveform = self.time_stretch(waveform)
            elif aug_name == "pitch_shift":
                waveform = self.pitch_shift(waveform)
            elif aug_name == "add_noise":
                waveform = self.add_noise(waveform)
            elif aug_name == "volume_change":
                waveform = self.volume_change(waveform)
            elif aug_name == "time_masking":
                waveform = self.time_masking(waveform)
                
        return waveform


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    (https://arxiv.org/abs/1904.08779)
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        replace_with_zero: bool = True,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.replace_with_zero = replace_with_zero
        
    def freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to spectrogram."""
        batch_size, n_mels, time_frames = spec.shape
        
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f_zero = random.randint(0, n_mels - f)
            
            if self.replace_with_zero:
                spec[:, f_zero:f_zero + f, :] = 0
            else:
                # Replace with mean value
                spec[:, f_zero:f_zero + f, :] = spec.mean()
                
        return spec
    
    def time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time masking to spectrogram."""
        batch_size, n_mels, time_frames = spec.shape
        
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_frames))
            t_zero = random.randint(0, time_frames - t)
            
            if self.replace_with_zero:
                spec[:, :, t_zero:t_zero + t] = 0
            else:
                # Replace with mean value
                spec[:, :, t_zero:t_zero + t] = spec.mean()
                
        return spec
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram."""
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec


class RandAugment:
    """
    RandAugment for audio, inspired by the vision version.
    Applies a fixed number of augmentations with varying magnitudes.
    """
    
    def __init__(
        self,
        n: int = 2,
        m: int = 10,
        sample_rate: int = 16000,
    ):
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude of augmentations (0-30)
        self.sample_rate = sample_rate
        
        # Initialize individual augmentations
        self.audio_aug = AudioAugmentation(sample_rate=sample_rate, augment_prob=1.0)
        
    def get_magnitude(self, name: str) -> Union[float, int]:
        """Get augmentation magnitude based on self.m."""
        magnitude_map = {
            "time_stretch": {
                "min_rate": 1.0 - 0.3 * (self.m / 30),
                "max_rate": 1.0 + 0.3 * (self.m / 30),
            },
            "pitch_shift": {
                "min_steps": -int(12 * (self.m / 30)),
                "max_steps": int(12 * (self.m / 30)),
            },
            "add_noise": {
                "min_noise": 0.0001,
                "max_noise": 0.02 * (self.m / 30),
            },
            "volume_change": {
                "min_gain_db": -30 * (self.m / 30),
                "max_gain_db": 20 * (self.m / 30),
            },
            "time_masking": {
                "mask_param": int(100 * (self.m / 30)),
            },
        }
        
        return magnitude_map.get(name, {})
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply RandAugment to waveform."""
        available_ops = [
            "time_stretch",
            "pitch_shift",
            "add_noise",
            "volume_change",
            "time_masking",
        ]
        
        # Randomly select n augmentations
        selected_ops = random.choices(available_ops, k=self.n)
        
        for op in selected_ops:
            params = self.get_magnitude(op)
            
            if op == "time_stretch":
                waveform = self.audio_aug.time_stretch(waveform, **params)
            elif op == "pitch_shift":
                waveform = self.audio_aug.pitch_shift(waveform, **params)
            elif op == "add_noise":
                waveform = self.audio_aug.add_noise(waveform, **params)
            elif op == "volume_change":
                waveform = self.audio_aug.volume_change(waveform, **params)
            elif op == "time_masking":
                waveform = self.audio_aug.time_masking(waveform, **params)
                
        return waveform