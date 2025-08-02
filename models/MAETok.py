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
from models.modules import Encoder, Decoder, HOGGeneratorMel
from models.softvq_vae import SoftVQModel, Normalize, Denormalize
from utils.model_utils import init_random_2d_freqs, init_t_xy, compute_axial_cis, compute_mixed_cis

try:
    from flash_attn import flash_attn_qkvpacked_func
except:
    flash_attn_qkvpacked_func = None

class MAETokConfig(PretrainedConfig):
    """Configuration class for MAETok model."""
    model_type = "MAETok"
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
        

class MAETokModel(SoftVQModel, PreTrainedModel):
    config_class = MAETokConfig
    def __init__(self, config: MAETokConfig):
        config.repa = True # for dinov2 decoder
        super().__init__(config)

        self.quantize = None
        self.aux_loss_mask = config.aux_loss_mask
        self.aux_hog_decoder = config.aux_hog_dec
        self.aux_dino_decoder = config.aux_dino_dec
        self.aux_clip_decoder = config.aux_clip_dec

        if self.aux_hog_decoder:
            print('Using HOG decoder: ', config.aux_dec_model)
            self.decoder_hog = Decoder(
                in_channels=1, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': (config.n_mels, config.max_time_frames), 
                              'patch_size': config.dec_patch_size, 
                              'drop_path_rate': 0.0, 
                              'latent_dim': config.codebook_embed_dim},
                pretrained = config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.dec_use_ape, 
                use_rope=config.dec_use_rope, 
                rope_mixed=config.dec_rope_mixed, 
                rope_theta=config.dec_rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_audio='identity',
                base_spec_size=config.base_spec_size
            )
            self.post_quant_conv_hog = nn.Linear(config.codebook_embed_dim, self.decoder_hog.embed_dim)
            self.to_audio_hog = nn.Linear(self.decoder_hog.embed_dim, 108)
            self.hog_generator = HOGGeneratorMel()
            if 'movq' in config.aux_dec_model:
                self.hog_use_movq = True 
            else:
                self.hog_use_movq = False
        
        
        if self.aux_dino_decoder:
            print('Using DINO decoder: ', config.aux_dec_model)
            self.decoder_dino = Decoder(
                in_channels=1, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': self.repa_model.img_size, 
                              'patch_size': self.repa_model.patch_size, 
                              'drop_path_rate': 0.0, 
                              'latent_dim': config.codebook_embed_dim},
                pretrained=False,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.dec_use_ape, 
                use_rope=config.dec_use_rope, 
                rope_mixed=config.dec_rope_mixed, 
                rope_theta=config.dec_rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_audio='identity',
                base_spec_size=config.base_spec_size
            )
            self.post_quant_conv_dino = nn.Linear(config.codebook_embed_dim, self.decoder_dino.embed_dim)
            self.to_audio_dino = nn.Linear(self.decoder_dino.embed_dim, self.repa_model.embed_dim)
            if 'movq' in config.aux_dec_model:
                self.dino_use_movq = True 
            else:
                self.dino_use_movq = False
        
        
        if self.aux_clip_decoder:
            self.clip_model = create_model('vit_so400m_patch14_siglip_gap_224', 
                                           pretrained=True, 
                                           img_size=(config.n_mels, config.max_time_frames), 
                                           patch_size=config.repa_patch_size)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            # self.clip_model.dynamic_img_size = True # Not sure if this is correct
            self.clip_model.eval()
            self.clip_de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.clip_scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            print('Using CLIP decoder: ', config.aux_dec_model)
            self.decoder_clip = Decoder(
                in_channels=1, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.aux_dec_model,
                model_kwargs={'img_size': self.clip_model.img_size, 
                              'patch_size': self.clip_model.patch_size, 
                              'drop_path_rate': 0.0, 
                              'latent_dim': config.codebook_embed_dim},
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.dec_use_ape, 
                use_rope=config.dec_use_rope, 
                rope_mixed=config.dec_rope_mixed, 
                rope_theta=config.dec_rope_theta,
                cls_token=config.aux_dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_audio='identity',
                base_spec_size=config.base_spec_size
            )
            self.post_quant_conv_clip = nn.Linear(config.codebook_embed_dim, self.decoder_clip.embed_dim)
            self.to_audio_clip = nn.Linear(self.decoder_clip.embed_dim, self.clip_model.embed_dim)
            if 'movq' in config.aux_dec_model:
                self.clip_use_movq = True 
            else:
                self.clip_use_movq = False
    
    def mean_flat(self, x):
        """
        Take the mean over all non-batch dimensions.
        """
        return torch.mean(x, dim=list(range(1, len(x.size()))))

    def encode(self, x):
        
        # h = self.encoder(x)
        if self.training:
            h, mask = self.encoder(x, return_mask=True)
        else:
            h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))
        info = None
        if self.training:
            return quant, emb_loss, info, mask
        else:
            return quant, emb_loss, info

    def decode_hog(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_hog(quant)
        if self.hog_use_movq:
            dec = self.decoder_hog(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_hog(quant, None, h, w)
        return dec
    
    def decode_dino(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_dino(quant)
        if self.dino_use_movq:
            dec = self.decoder_dino(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_dino(quant, None, h, w)
        return dec

    def decode_clip(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_clip(quant)
        if self.clip_use_movq:
            dec = self.decoder_clip(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_clip(quant, None, h, w)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        # print('input shape: ', input.shape)
        # encode
        if self.training:
            quant, diff, info, mask = self.encode(input)
        else:
            quant, diff, info = self.encode(input)
        self.quant = quant
        # print(quant.shape)
        dec = self.decode(quant, x=input, h=h, w=w)
         
        # decode hog
        if self.training:
            # decode hog feature
            if self.aux_hog_decoder:
                dec_hog = self.decode_hog(quant, x=input, h=h, w=w)   
                dec_hog = self.to_audio_hog(dec_hog)
                # get hog_target
                z_hog = self.hog_generator(input) 
                if self.aux_loss_mask:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog, reduction='none')
                    hog_rec_loss = (hog_rec_loss * mask).sum() / mask.sum() / z_hog.size(-1)
                else:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog)
            else:
                hog_rec_loss = 0.0
        
            # decode dinov2 feature
            if self.aux_dino_decoder:
                dec_dino = self.decode_dino(quant, x=input, h=h, w=w)
                dec_dino = self.to_audio_dino(dec_dino)
                
                # get z from repa_encoder
                rescale_x = self.scale(self.de_scale(input))
                z_dino = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

                z_dino = F.normalize(z_dino, dim=-1)
                dec_dino = F.normalize(dec_dino, dim=-1)

                if self.aux_loss_mask:
                    dino_rec_loss = -(dec_dino * z_dino).sum(dim=-1, keepdim=True)
                    dino_rec_loss = (dino_rec_loss * mask).sum() / mask.sum()
                else:
                    dino_rec_loss = self.mean_flat(-(dec_dino * z_dino).sum(dim=-1))
                    dino_rec_loss = dino_rec_loss.mean()
            else:
                dino_rec_loss = 0.0
            
            # deocde clip feature
            if self.aux_clip_decoder:
                dec_clip = self.decode_clip(quant, x=input, h=h, w=w)
                dec_clip = self.to_audio_clip(dec_clip)
                # get clip_target
                rescale_x = self.clip_scale(self.clip_de_scale(input))
               
                z_clip = self.clip_model.forward_features(rescale_x)[:, self.clip_model.num_prefix_tokens:]
                
                z_clip = F.normalize(z_clip, dim=-1)
                dec_clip = F.normalize(dec_clip, dim=-1)
                
                if self.aux_loss_mask:
                    clip_rec_loss = -(dec_clip * z_clip).sum(dim=-1, keepdim=True)
                    clip_rec_loss = (clip_rec_loss * mask).sum() / mask.sum()
                else:
                    clip_rec_loss = self.mean_flat(-(dec_clip * z_clip).sum(dim=-1))
                    clip_rec_loss = clip_rec_loss.mean()   
            else:
                clip_rec_loss = 0.0
            
            diff += (dino_rec_loss, hog_rec_loss, clip_rec_loss, )

        return dec, diff, info

if __name__ == "__main__":
    config = MAETokConfig.from_pretrained("configs/MAETok.json")
    model = MAETokModel(config)
    model.train()

    dummy_input = torch.randn(2, 1, 128, 1000)  # Batch size of 2, 1 channel, 128 frequency bins, 1000 time frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input = dummy_input.to(device)
    output, diff, info = model(dummy_input)