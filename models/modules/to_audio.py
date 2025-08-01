import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class ToAudio(nn.Module):
    def __init__(self, to_audio='linear', spec_size=(128, 1000), in_channels=1, in_dim=512, patch_freq=16, patch_time=8) -> None:
        super().__init__()
        self.to_audio_name = to_audio
        self.spec_size = spec_size
        self.in_channels = in_channels
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.num_patches = (spec_size[0] // patch_freq) * (spec_size[1] // patch_time)
        if to_audio == 'linear':
            self.model = nn.Linear(in_dim, in_channels * patch_freq * patch_time)
        elif to_audio == 'conv':
            num_patches_per_dim = spec_size[0] // patch_freq  # e.g. 256//16 = 16
            self.model = nn.Sequential(
                # (B, L, C) -> (B, C, H, W) with H = W = num_patches_per_dim
                Rearrange('b (h w) c -> b c h w', h=num_patches_per_dim),
                
                # For example, first reduce dimension via a 1x1 conv from in_dim -> 128
                nn.Conv2d(in_dim, 128, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),

                # Upsample from size (num_patches_per_dim) to a larger intermediate
                nn.Upsample(scale_factor=2, mode='nearest'),  
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                # Repeat upsampling until we reach the final resolution
                # For a 16x16 patch layout, we need 4x upsampling to reach 256
                #   16 -> 32 -> 64 -> 128 -> 256
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
            )
        elif to_audio == 'siren':
            self.model = nn.Sequential(
                SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                SineLayer(in_dim * 2, spec_size[0] // patch_freq * patch_freq * in_channels, is_first=False, omega_0=30)
            )
        elif to_audio == 'identity':
            self.model = nn.Identity()
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_audio_name == 'linear':
            return self.model.weight
        elif self.to_audio_name == 'siren':
            return self.model[1].linear.weight
        elif self.to_audio_name == 'conv':
            return self.model[-1].weight
        else:
            return None

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        对应图像版 unpatchify 的声谱图版本
        Args
        ----
        x : (B, L, C * pf * pt)
            L  =  (F // pf) * (T // pt)
        Returns
        -------
        spec : (B, C, F, T)
        """
        B, L, _ = x.shape
        C, pf, pt = self.in_channels, self.patch_freq, self.patch_time
        F,  T     = self.spec_size

        # 计算补丁网格尺寸
        h = F // pf          # 补丁在频率方向的个数
        w = T // pt          # 补丁在时间方向的个数
        assert h * w == L, f"token 数不匹配: expected {h*w}, got {L}"

        # (B, h*w, C·pf·pt)  →  (B, h, w, pf, pt, C)
        x = x.reshape(B, h, w, pf, pt, C)

        # 把 C 维提前，并把小 patch 拼回完整频率和时间分辨率
        # 维度次序: B, C, h, pf, w, pt
        x = x.permute(0, 5, 1, 3, 2, 4)          # (B, C, h, pf, w, pt)
        spec = x.reshape(B, C, h*pf, w*pt)       # (B, C, F, T)

        return spec

    def forward(self, x):
        if self.to_audio_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x)
        elif self.to_audio_name == 'siren':
            x = self.model(x)
            x = x.view(x.shape[0], self.in_channels, self.patch_size * int(self.num_patches ** 0.5),
                       self.patch_size * int(self.num_patches ** 0.5))
        elif self.to_audio_name == 'conv':
            x = self.model(x)
        elif self.to_audio_name == 'identity':
            pass
        return x