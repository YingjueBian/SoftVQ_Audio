from typing import Tuple, List

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm

from einops import rearrange

from collections import namedtuple


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
        num_embeddings: int = None,
        expect_spectrum: bool = True,   # NEW
    ):
        super().__init__()
        self.expect_spectrum = expect_spectrum
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(r, num_embeddings=num_embeddings, expect_spectrum=expect_spectrum)
             for r in resolutions]
        )

    def forward(self, y_spec, y_spec_hat, bandwidth_id=None):
        # y_spec/y_hat_spec: (B, F, T)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y_spec, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_spec_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution,
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
        lrelu_slope: float = 0.1,
        expect_spectrum: bool = True,    # NEW
    ):
        super().__init__()
        self.resolution = resolution
        self.expect_spectrum = expect_spectrum
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2))),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1))),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1)),
            weight_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1)),
        ])
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(num_embeddings, channels)
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x, cond_embedding_id=None):
        """
        x: 如果 expect_spectrum=True, shape=(B, F, T)
           否则传 waveform, 内部会转谱（保持兼容）
        """
        fmap = []

        if not self.expect_spectrum:
            x = self._spectrogram_old(x)  # 原始 torch.stft 实现
        # x: (B, F, T)
        # x = x.unsqueeze(1)               # -> (B, 1, F, T)

        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)

        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0

        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def _spectrogram_old(self, x):
        n_fft, hop_length, win_length = self.resolution
        return torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=None, center=True, return_complex=True
        ).abs()
    


