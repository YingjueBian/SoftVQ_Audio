import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple, Union
import numpy as np


class AudioFeatureExtractor:
    """Simplified audio feature extractor using torchaudio."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalize: bool = True,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = "slaney",
        mel_scale: str = "htk",
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalize = normalize
        
        # Initialize mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            pad_mode=pad_mode,
            n_mels=n_mels,
            center=center,
            power=power,
            norm=norm,
            mel_scale=mel_scale,
        )
        
        # Amplitude to dB transform
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        
    def __call__(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract mel-spectrogram features from audio.
        
        Args:
            audio: Audio waveform tensor of shape (batch, time) or (time,)
            sample_rate: Sample rate of the audio. If provided and different from
                        self.sample_rate, will resample.
                        
        Returns:
            Mel-spectrogram tensor of shape (batch, n_mels, time_frames)
        """
        # Convert numpy to torch if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
            
        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Resample if needed
        if sample_rate and sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            audio = resampler(audio)
            
        # Extract mel-spectrogram
        mel_spec = self.mel_spectrogram(audio)
        
        # Convert to log scale
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Normalize if requested
        if self.normalize:
            # Per-channel normalization
            mean = log_mel_spec.mean(dim=-1, keepdim=True)
            std = log_mel_spec.std(dim=-1, keepdim=True)
            log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
            
        return log_mel_spec
    
    def get_frames_per_second(self) -> float:
        """Get the number of frames per second in the mel-spectrogram."""
        return self.sample_rate / self.hop_length


class AudioPatchExtractor:
    """Extract patches from mel-spectrograms for vision transformer-style processing."""
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        flatten: bool = True,
    ):
        self.patch_height, self.patch_width = patch_size
        self.embed_dim = embed_dim
        self.flatten = flatten
        
        # Calculate patch embedding dimension
        self.patch_dim = self.patch_height * self.patch_width
        
        # Linear projection for patches
        self.projection = torch.nn.Linear(self.patch_dim, embed_dim)
        
    def extract_patches(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram of shape (batch, n_mels, time_frames)
            
        Returns:
            Patches of shape (batch, num_patches, patch_height, patch_width) if not flattened
            or (batch, num_patches, patch_dim) if flattened
        """
        batch_size, n_mels, time_frames = mel_spec.shape
        
        # Calculate number of patches
        n_patches_freq = n_mels // self.patch_height
        n_patches_time = time_frames // self.patch_width
        
        # Reshape to extract patches
        patches = mel_spec[:, :n_patches_freq * self.patch_height, :n_patches_time * self.patch_width]
        patches = patches.reshape(
            batch_size,
            n_patches_freq, self.patch_height,
            n_patches_time, self.patch_width
        )
        patches = patches.permute(0, 1, 3, 2, 4)  # (B, n_patch_freq, n_patch_time, patch_h, patch_w)
        patches = patches.reshape(batch_size, n_patches_freq * n_patches_time, self.patch_height, self.patch_width)
        
        if self.flatten:
            patches = patches.reshape(batch_size, -1, self.patch_dim)
            
        return patches
    
    def __call__(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract and project patches from mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram of shape (batch, n_mels, time_frames)
            
        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        patches = self.extract_patches(mel_spec)
        
        if self.flatten:
            # Project flattened patches to embedding dimension
            patch_embeds = self.projection(patches)
        else:
            # Flatten then project
            batch_size, num_patches = patches.shape[:2]
            patches_flat = patches.reshape(batch_size, num_patches, -1)
            patch_embeds = self.projection(patches_flat)
            
        return patch_embeds


def pad_or_truncate_audio(
    audio: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad or truncate audio to target length.
    
    Args:
        audio: Audio tensor of shape (batch, time) or (time,)
        target_length: Target length in samples
        pad_value: Value to use for padding
        
    Returns:
        Audio tensor of shape (batch, target_length) or (target_length,)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
        
    batch_size, current_length = audio.shape
    
    if current_length == target_length:
        output = audio
    elif current_length < target_length:
        # Pad
        padding = target_length - current_length
        output = torch.nn.functional.pad(audio, (0, padding), value=pad_value)
    else:
        # Truncate
        output = audio[:, :target_length]
        
    if squeeze_output:
        output = output.squeeze(0)
        
    return output


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using torchaudio.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize audio to [-1, 1]
        
    Returns:
        Tuple of (audio_tensor, original_sample_rate)
    """
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if needed
    if orig_sr != sample_rate:
        resampler = T.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)
        
    # Normalize if requested
    if normalize:
        waveform = waveform / (waveform.abs().max() + 1e-6)
        
    return waveform.squeeze(0), orig_sr


def extract_mel_spectrogram(
    audio: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    target_sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    max_duration: float = 10.0,
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Optimized mel-spectrogram extraction for training.
    
    Args:
        audio: Audio waveform
        sample_rate: Original sample rate
        target_sample_rate: Target sample rate
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop size
        win_length: Window size
        f_min: Minimum frequency
        f_max: Maximum frequency
        max_duration: Maximum duration in seconds
        normalize: Whether to normalize
        device: Device to use for computation
        
    Returns:
        Mel-spectrogram tensor of shape (1, n_mels, time_frames)
    """
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # Move to device if specified
    if device is not None:
        audio = audio.to(device)
    
    # Ensure 1D audio
    if audio.dim() > 1:
        audio = audio.squeeze()
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate).to(audio.device)
        audio = resampler(audio)
    
    # Pad or truncate to max duration
    max_samples = int(max_duration * target_sample_rate)
    audio = pad_or_truncate_audio(audio, max_samples)
    
    # Create mel spectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max or target_sample_rate // 2,
        n_mels=n_mels,
        norm="slaney",
        mel_scale="htk",
    ).to(audio.device)
    
    # Extract mel spectrogram
    mel_spec = mel_transform(audio.unsqueeze(0))
    
    # Convert to log scale
    log_mel_spec = torch.log(mel_spec + 1e-10)
    
    # Normalize
    if normalize:
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
    
    return log_mel_spec