import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as tdist
import numpy as np
import peft
import math
import scipy.stats as stats
from functools import partial

from einops.layers.torch import Rearrange

from timm import create_model
from timm.layers import trunc_normal_
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked
from transformers import PretrainedConfig, PreTrainedModel
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional, Union, Callable, Sequence, Literal
# from models.vision_transformer_softvq import Attention, MoVQNorm, MoVQBlockv2
from models.quantizer import SoftVectorQuantizer
from models.modules import Encoder, Decoder
from utils.model_utils import init_random_2d_freqs, init_t_xy, compute_axial_cis, compute_mixed_cis

try:
    from flash_attn import flash_attn_qkvpacked_func
except:
    flash_attn_qkvpacked_func = None

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean

class SoftVQConfig(PretrainedConfig):
    """Configuration class for SoftVQ model."""
    model_type = "softvq"
    def __init__(self,
        # Audio parameters
        sample_rate: int = 16000,
        n_mels: int = 128,
        patch_size: Tuple[int, int] = (16, 16),
        speech_patch_dim: int = 256,
        max_time_frames: int = 1000,
        base_spec_size: Tuple[int, int] = (128, 1000),

        # Tokenizer parameters
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        codebook_l2_norm: bool = True,
        codebook_show_usage: bool = False,
        commit_loss_beta: float = 0.25,
        tau: float = 0.1,
        num_codebooks: int = 1,

        vq_mean: float = 0.0,
        vq_std: float = 1.0,

        entropy_loss_ratio: float = 0.0,
        vq_loss_ratio: float = 1.0, # for soft vq
        kl_loss_weight: float = 0.000001,

        # model parameters
        encoder_ch_mult: Optional[List[int]] = None,
        decoder_ch_mult: Optional[List[int]] = None,

        dropout_p: float = 0.0,

        # enc_type: str = 'vit',
        # dec_type: str = 'vit',
        encoder_model: str = 'llamagen_encoder',
        decoder_model: str = 'llamagen_decoder',
        num_latent_tokens: int = 256,

        enc_patch_size: Tuple[int, int] = (16, 16),
        dec_patch_size: Tuple[int, int] = (16, 16),
        enc_drop_path_rate: float = 0.0,
        dec_drop_path_rate: float = 0.0,
        to_audio: str = 'linear',

        enc_tuning_method: str = 'full',
        dec_tuning_method: str = 'full',
        enc_pretrained: bool = True,
        dec_pretrained: bool = False,

        # rope
        enc_use_ape: bool = True,
        enc_use_rope: bool = False,
        dec_use_ape: bool = True,
        dec_use_rope: bool = False,
        enc_rope_mixed: bool = False,
        enc_rope_theta: float = 10.0,
        dec_rope_mixed: bool = False,
        dec_rope_theta: float = 10.0,

        # repa for vit
        repa: bool = False,
        repa_patch_size: Tuple[int, int] = (16, 16),
        repa_model: str = 'vit_base_patch16_224',
        repa_proj_dim: int = 2048,
        repa_loss_weight: float = 1.0,
        repa_align: str = 'global',

        # encoder token drop for mask modeling
        enc_token_drop: float = 0.0,
        enc_token_drop_max: float = 0.6,
        
        dec_cls_token: bool = True,

        # auxdecoder model
        aux_dec_model: str = 'vit_tiny_patch14_dinov2_movq',
        aux_loss_mask: bool = False,
        aux_dec_cls_token: bool = True,
        aux_hog_dec: bool = True,
        aux_dino_dec: bool = True,
        aux_supcls_dec: bool = True,
        aux_clip_dec: bool = True,
        **kwargs,
        ):  
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.speech_patch_dim = speech_patch_dim
        self.max_time_frames = max_time_frames
        self.base_spec_size = base_spec_size
        
        # codebook
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.codebook_l2_norm = codebook_l2_norm
        self.codebook_show_usage = codebook_show_usage
        self.commit_loss_beta = commit_loss_beta
        self.tau = tau
        self.num_codebooks = num_codebooks
        self.vq_mean = vq_mean
        self.vq_std = vq_std
        self.entropy_loss_ratio = entropy_loss_ratio
        self.vq_loss_ratio = vq_loss_ratio
        self.kl_loss_weight = kl_loss_weight

        # model parameters
        self.encoder_ch_mult = encoder_ch_mult
        self.decoder_ch_mult = decoder_ch_mult
        self.dropout_p = dropout_p

        # encoder
        self.encoder_model = encoder_model
        self.num_latent_tokens = num_latent_tokens
        self.enc_patch_size = enc_patch_size
        self.enc_drop_path_rate = enc_drop_path_rate
        self.enc_pretrained = enc_pretrained
        self.enc_tuning_method = enc_tuning_method
        self.enc_use_ape = enc_use_ape
        self.enc_use_rope = enc_use_rope

        # decoder
        self.decoder_model = decoder_model
        self.dec_patch_size = dec_patch_size
        self.dec_drop_path_rate = dec_drop_path_rate
        self.dec_pretrained = dec_pretrained
        self.dec_tuning_method = dec_tuning_method
        self.dec_use_ape = dec_use_ape
        self.dec_use_rope = dec_use_rope
        self.dec_cls_token = dec_cls_token

        # repa
        self.repa = repa
        self.repa_loss_weight = repa_loss_weight
        self.repa_align = repa_align
        self.repa_model = repa_model
        self.repa_proj_dim = repa_proj_dim
        self.repa_patch_size = repa_patch_size

        # encoder token drop
        self.enc_token_drop = enc_token_drop
        self.enc_token_drop_max = enc_token_drop_max

        # auxdecoder model
        self.aux_dec_model = aux_dec_model
        self.aux_loss_mask = aux_loss_mask
        self.aux_dec_cls_token = aux_dec_cls_token
        self.aux_hog_dec = aux_hog_dec
        self.aux_dino_dec = aux_dino_dec
        self.aux_supcls_dec = aux_supcls_dec
        self.aux_clip_dec = aux_clip_dec

        # rope
        self.enc_rope_mixed = enc_rope_mixed
        self.enc_rope_theta = enc_rope_theta

        self.dec_rope_mixed = dec_rope_mixed
        self.dec_rope_theta = dec_rope_theta

        self.to_audio = to_audio

class SoftVQModel(PreTrainedModel):
    config_class = SoftVQConfig

    def __init__(self, config: SoftVQConfig):
        super().__init__(config)
        self.config = config
        
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        self.repa_proj_dim = config.repa_proj_dim
        if config.repa:
            self.repa_model = create_model(config.repa_model, pretrained=True, 
                                           img_size=(config.n_mels,config.max_time_frames), 
                                           patch_size=config.repa_patch_size)
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            repa_z_dim = self.repa_model.embed_dim
            self.repa_z_dim = repa_z_dim

            self.projection = nn.Sequential(
                nn.Linear(self.codebook_embed_dim, self.repa_proj_dim),
                nn.SiLU(),
                nn.Linear(self.repa_proj_dim, self.repa_proj_dim),
                nn.SiLU(),
                nn.Linear(self.repa_proj_dim, self.repa_z_dim),
            )
            # from .lpips_timm import Normalize, Denormalize
            self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.repa_z_dim = None
        
        self.encoder = Encoder(in_channels=1, 
                               num_latent_tokens=config.num_latent_tokens,
                               model_name= config.encoder_model,
                               model_kwargs={
                                    'img_size': (config.n_mels, config.max_time_frames),
                                    'patch_size': config.enc_patch_size,
                                    'drop_path_rate': config.enc_drop_path_rate,
                                    'in_chans': 1,
                                   },
                                pretrained=config.enc_pretrained,
                                tuning_method=config.enc_tuning_method,
                                tuning_kwargs={'r': 8},
                                use_ape=config.enc_use_ape,
                                use_rope=config.enc_use_rope,
                                rope_mixed=config.enc_rope_mixed,
                                rope_theta=config.enc_rope_theta,
                                token_drop=config.enc_token_drop,
                                token_drop_max=config.enc_token_drop_max,
                                base_spec_size=config.base_spec_size,
                                )
                                
        self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)

        self.decoder = Decoder(in_channels=1, 
                               num_latent_tokens=config.num_latent_tokens,
                               model_name=config.decoder_model,
                               model_kwargs={
                                    'img_size': (config.n_mels, config.max_time_frames),
                                    'patch_size': config.dec_patch_size,
                                    'drop_path_rate': config.dec_drop_path_rate,
                                    'in_chans': 1,
                                    'latent_dim': config.codebook_embed_dim,},
                                pretrained=config.dec_pretrained,
                                tuning_method=config.dec_tuning_method,
                                tuning_kwargs={'r': 8},
                                use_ape=config.dec_use_ape,
                                use_rope=config.dec_use_rope,
                                rope_mixed=config.dec_rope_mixed,
                                rope_theta=config.dec_rope_theta,
                                cls_token=config.dec_cls_token,
                                codebook_embed_dim=config.codebook_embed_dim,
                                to_audio=config.to_audio,
                                base_spec_size=config.base_spec_size,
                               )
        self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False
        
        self.quantize = SoftVectorQuantizer(config.codebook_size, 
                                            config.codebook_embed_dim, 
                                            config.entropy_loss_ratio, 
                                            config.tau,                                   
                                            config.num_codebooks,
                                            config.codebook_l2_norm, 
                                            config.codebook_show_usage)
    def mean_flat(self, x):
        """
        Take the mean over all non-batch dimensions.
        """
        return torch.mean(x, dim=list(range(1, len(x.size()))))

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            # get z from repa_encoder
            rescale_x = self.scale(self.de_scale(x))
            z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = self.mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w)
        return dec, diff, info

if __name__ == "__main__":
    config = SoftVQConfig.from_pretrained("configs/softvq_config.json")
    model = SoftVQModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # Test the model with a dummy input
    dummy_input = torch.randn(10, 1, 128, 1000)  # Batch size of 10, 1 channel, 128x1000 image
    dummy_input = dummy_input.to(device)
    output, diff, info = model(dummy_input)
