import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HOGGeneratorMel(nn.Module):
    def __init__(self, nbins=9, pool=8, gaussian_window=16):
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi

        # Sobel filter for time (x axis) and freq (y axis)
        weight_t = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_t = weight_t.view(1, 1, 3, 3).contiguous()  # For time axis gradient
        weight_f = weight_t.transpose(2, 3).contiguous()   # For freq axis gradient
        self.register_buffer('weight_t', weight_t)
        self.register_buffer('weight_f', weight_f)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gaussian_kernel = self.get_gaussian_kernel(gaussian_window, gaussian_window // 2)
            self.register_buffer('gaussian_kernel', gaussian_kernel)

    def get_gaussian_kernel(self, kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n ** 2)
        kernel_2d = w[:, None] * w[None, :]
        return kernel_2d / kernel_2d.sum()

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: [B, 1, F, T] mel spectrogram
        Returns:
            HOG features: [B, *, nbins]
        """
        B, C, F_, T_ = x.shape
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')  # pad time and freq

        gx = F.conv2d(x, self.weight_t, stride=1, padding=0)
        gy = F.conv2d(x, self.weight_f, stride=1, padding=0)
        grad_mag = torch.stack([gx, gy], dim=-1).norm(dim=-1)  # [B, 1, F, T]
        phase = torch.atan2(gx, gy) / self.pi * self.nbins

        # Shape to [B, 1, nbins, F, T]
        out = torch.zeros((B, 1, self.nbins, F_, T_), dtype=torch.float, device=x.device)
        phase = phase.view(B, 1, 1, F_, T_)
        grad_mag = grad_mag.view(B, 1, 1, F_, T_)

        if self.gaussian_window:
            if F_ != self.gaussian_window:
                assert F_ % self.gaussian_window == 0
                repeat_rate = F_ // self.gaussian_window
                temp_gaussian_kernel = self.gaussian_kernel.repeat(repeat_rate, repeat_rate)
            else:
                temp_gaussian_kernel = self.gaussian_kernel
            grad_mag *= temp_gaussian_kernel

        out.scatter_add_(2, phase.floor().long() % self.nbins, grad_mag)

        # Pool over local patches
        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])  # sum over patch
        out = F.normalize(out, p=2, dim=2)

        # Reshape to [B, NumPatches, nbins]
        B, C, nbins, nF, nT = out.shape
        return out.permute(0, 3, 4, 2).reshape(B, -1, nbins)

