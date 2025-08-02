import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2

class HOGGeneratorMel(nn.Module):
    def __init__(self, nbins=108, pool_freq=16, pool_time=25, gaussian_window=16,
                 target_num_patches=None, target_dim=None):
        """
        Args:
            nbins: number of histogram bins
            pool_freq: pooling size in frequency (height) dimension
            pool_time: pooling size in time (width) dimension
            gaussian_window: size of gaussian window for smoothing
            target_num_patches: if set, interpolates output to this number of patches
            target_dim: if set, projects feature dimension to this value
        """
        super().__init__()
        self.nbins = nbins
        self.pool_freq = pool_freq
        self.pool_time = pool_time
        self.pi = math.pi
        self.target_num_patches = target_num_patches
        self.target_dim = target_dim

        # Sobel filters
        sobel_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # time
        self.register_buffer('weight_t', sobel_x.view(1, 1, 3, 3))  # for T
        self.register_buffer('weight_f', sobel_x.T.view(1, 1, 3, 3))  # for F

        # Gaussian kernel
        self.gaussian_window = gaussian_window
        if gaussian_window:
            kernel = self.get_gaussian_kernel(gaussian_window, gaussian_window // 2)
            self.register_buffer('gaussian_kernel', kernel)

        if target_dim is not None:
            self.projector = nn.Linear(nbins, target_dim)

    def get_gaussian_kernel(self, kernlen: int, std: int):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        kernel_2d = w[:, None] * w[None, :]
        return kernel_2d / kernel_2d.sum()

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: [B, 1, F, T]
        Returns:
            [B, N_patches, nbins] (or [B, N_patches, target_dim] if projection applied)
        """
        B, C, F_, T_ = x.shape
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')

        gx = F.conv2d(x, self.weight_t, padding=0)
        gy = F.conv2d(x, self.weight_f, padding=0)
        grad_mag = torch.stack([gx, gy], dim=-1).norm(dim=-1)  # [B, 1, F, T]

        phase = torch.atan2(gx, gy) / self.pi * self.nbins
        phase_bin = (phase.floor().long() % self.nbins).unsqueeze(2)   # [B, 1, 1, F, T]

        grad_mag = grad_mag.unsqueeze(2)     

        out = torch.zeros(B, 1, self.nbins, F_, T_, device=x.device)
        out.scatter_add_(2, phase_bin, grad_mag)

        # Apply Gaussian smoothing
        if self.gaussian_window:
            gk = self.gaussian_kernel
            gk = F.interpolate(gk[None, None], size=(F_, T_), mode='bilinear', align_corners=False)
            grad_mag *= gk.squeeze(0).squeeze(0)

        # Pooling
        out = out.unfold(3, self.pool_freq, self.pool_freq)
        out = out.unfold(4, self.pool_time, self.pool_time)  # [B, 1, nbins, nF, nT, pf, pt]
        out = out.sum(dim=[-1, -2])  # sum over patch window
        out = F.normalize(out, p=2, dim=2)  # [B, 1, nbins, nF, nT]
        out = out.squeeze(1).permute(0, 2, 3, 1).reshape(B, -1, self.nbins)  # [B, nF*nT, nbins]

        # Optional: interpolate to target patch number
        if self.target_num_patches and out.shape[1] != self.target_num_patches:
            out = F.interpolate(out.transpose(1, 2), size=self.target_num_patches, mode='linear').transpose(1, 2)

        # Optional: project to final dim
        if self.target_dim:
            out = self.projector(out)

        return out