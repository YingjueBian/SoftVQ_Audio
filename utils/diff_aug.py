import torch
import torch.nn.functional as F
import random

# ---------- 单通道友好版增强函数 ----------
def rand_gain(x, gain_range=(0.8, 1.2)):
    g = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(*gain_range)
    return x * g

def rand_bias(x, bias_range=(-0.1, 0.1)):
    b = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(*bias_range)
    return x + b

def rand_time_shift(x, shift_max=80):
    """沿时间轴循环移位"""
    if shift_max <= 0: 
        return x
    B, C, F, T = x.shape
    shift = torch.randint(-shift_max, shift_max + 1, (B,), device=x.device)
    rolled = torch.stack([torch.roll(x[i], shifts=int(s), dims=-1) for i, s in enumerate(shift)])
    return rolled

def cutout_2d(x, freq_frac=0.2, time_frac=0.2):
    B, C, F, T = x.shape
    f = int(F * freq_frac)
    t = int(T * time_frac)
    for i in range(B):
        f0 = random.randint(0, F - f)
        t0 = random.randint(0, T - t)
        x[i, :, f0:f0+f, t0:t0+t] = 0
    return x

def specaugment_time_mask(x, num_masks=2, mask_size=80):
    B, C, F, T = x.shape
    for _ in range(num_masks):
        t = random.randint(0, T - mask_size)
        x[:, :, :, t:t+mask_size] = 0
    return x

# ---------- 构建 Mel-friendly AUGMENT_FNS ----------
AUGMENT_FNS_MEL = {
    "gain":   [rand_gain],
    "bias":   [rand_bias],
    "shift":  [rand_time_shift],
    "cutout": [cutout_2d],
    "tmask":  [specaugment_time_mask],
}

# ---------- 改写 DiffAugment ----------
def DiffAugment(x, policy="", prob=0.5):
    """
    x: Tensor [B, 1, n_mels, T]
    policy: e.g. "gain,shift,cutout,tmask"
    """
    if not policy:
        return x

    for p in policy.split(","):
        p = p.strip()
        if p not in AUGMENT_FNS_MEL:
            raise ValueError(f"Unknown policy '{p}' for mel spectrum")
        if torch.rand(1, device=x.device) > prob:
            continue
        for f in AUGMENT_FNS_MEL[p]:
            x = f(x)
    return x.contiguous()