import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(
    model: nn.Module,
    layers_to_freeze: Optional[List[str]] = None,
    freeze_embeddings: bool = True,
):
    """
    Freeze specific layers in the model.
    
    Args:
        model: The model to freeze layers in
        layers_to_freeze: List of layer names to freeze. If None, no layers are frozen.
        freeze_embeddings: Whether to freeze embedding layers
    """
    if layers_to_freeze is None:
        layers_to_freeze = []
        
    for name, param in model.named_parameters():
        # Freeze embeddings if requested
        if freeze_embeddings and "embed" in name:
            param.requires_grad = False
            
        # Freeze specific layers
        for layer_name in layers_to_freeze:
            if layer_name in name:
                param.requires_grad = False
                
                
def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return size_mb


def save_model_config(config: Dict, save_path: str):
    """Save model configuration to JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
        
        
def load_model_config(config_path: str) -> Dict:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def compute_output_length(
    input_length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    Compute output length for 1D convolution or pooling.
    
    Args:
        input_length: Length of input sequence
        kernel_size: Size of the kernel
        stride: Stride of the operation
        padding: Padding size
        dilation: Dilation factor
        
    Returns:
        Output length
    """
    return int((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def compute_mel_spec_length(
    audio_length_seconds: float,
    sample_rate: int,
    hop_length: int,
) -> int:
    """
    Compute the length of mel-spectrogram time dimension.
    
    Args:
        audio_length_seconds: Audio length in seconds
        sample_rate: Sample rate
        hop_length: Hop length for STFT
        
    Returns:
        Number of time frames in mel-spectrogram
    """
    audio_length_samples = int(audio_length_seconds * sample_rate)
    return audio_length_samples // hop_length + 1


def compute_num_patches(
    mel_bins: int,
    time_frames: int,
    patch_size: Tuple[int, int],
) -> int:
    """
    Compute number of patches from mel-spectrogram dimensions.
    
    Args:
        mel_bins: Number of mel frequency bins
        time_frames: Number of time frames
        patch_size: (height, width) of each patch
        
    Returns:
        Number of patches
    """
    patch_height, patch_width = patch_size
    n_patches_freq = mel_bins // patch_height
    n_patches_time = time_frames // patch_width
    return n_patches_freq * n_patches_time


def create_sinusoidal_embeddings(
    n_pos: int,
    dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Create sinusoidal position embeddings.
    
    Args:
        n_pos: Number of positions
        dim: Embedding dimension
        base: Base for the sinusoidal functions
        
    Returns:
        Position embeddings of shape (n_pos, dim)
    """
    position = torch.arange(n_pos).unsqueeze(1)
    dim_indices = torch.arange(0, dim, 2)
    div_term = torch.exp(dim_indices * -(np.log(base) / dim))
    
    embeddings = torch.zeros(n_pos, dim)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    
    return embeddings


def apply_rotary_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim)
        key: Key tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim)
        sin: Sine embeddings of shape (seq_len, head_dim)
        
    Returns:
        Tuple of (rotated_query, rotated_key)
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    # Apply rotary embeddings
    query_rot = query * cos + rotate_half(query) * sin
    key_rot = key * cos + rotate_half(key) * sin
    
    return query_rot, key_rot


class EMA:
    """
    Exponential Moving Average for model parameters.
    Useful for training stability and better generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.decay = decay
        self.device = device
        
        # Create EMA parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if self.device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
                    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_val = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_val)
                
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
                
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
                
        self.backup = {}


def get_activation_fn(activation: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }
    
    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")
        
    return activations[activation]

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def compute_mixed_cis(freqs, t_x, t_y, num_heads):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)