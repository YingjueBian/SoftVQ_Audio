import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import scipy.stats as stats

import peft
from timm.models import create_model
from timm.layers import trunc_normal_
from .to_audio import ToAudio
from .vision_transformer import Attention, MoVQNorm, MoVQBlockv2
from .rope_utils import compute_axial_cis, compute_mixed_cis, init_random_2d_freqs, init_t_xy



def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class Encoder(nn.Module):
    def __init__(self, in_channels=1, 
                 num_latent_tokens=32,
                 model_name = 'vit_small_patch14_dinov2.lvd142m',
                 model_kwargs = None,
                 pretrained = True,
                 tuning_method = 'full',
                 tuning_kwargs = None,
                 rope_theta = 10000,
                 rope_mixed = False,
                 use_rope = False,
                 use_ape = True,
                 token_drop = 0.4,
                 token_drop_max = 0.6,
                 base_spec_size = (128, 1000),
                config = None
                 ):
        super().__init__()
        

        self.model_name = model_name
        self.config = config
        # assert self.model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
        #                       'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
        #                       'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
        #                       'vit_large_patch14_reg4_dinov2.lvd142m',
        #                       'vit_giant_patch14_reg4_dinov2.lvd142m', 'vit_base_patch16_clip_224.openai',
        #                       "vit_base_patch16_clip_224.laion2b", "samvit_base_patch16.sa1b", "eva02_base_patch16_clip_224.merged2b"], f"{self.model_name} not found"

        # load model

        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )

        self.n_mels = model_kwargs['img_size'][0]
        self.max_time_frames = model_kwargs['img_size'][1]
        F, T = self.n_mels, self.max_time_frames
        self.patch_size = model_kwargs['patch_size']
        pf, pt = self.patch_size
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_aud_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        
        # tuning method
        if tuning_method == 'full':
            self.model = model
        elif tuning_method == 'lora':
            Lora_config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['norm'], r = 8)
            self.model = peft.get_peft_model(model, Lora_config)
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
        else:
            raise ValueError(f"Unknown tuning method: {tuning_method}")

        # parameters
        self.num_latent_tokens = num_latent_tokens
        if self.num_latent_tokens:
            # latent tokens
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            nn.init.normal_(self.latent_tokens, std=.02)

            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            trunc_normal_(self.latent_pos_embed, std=.02)

        # token drop
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
            nn.init.normal_(self.mask_token, std=.02)

        # rope
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta

        if self.rope_mixed and self.use_rope:
            self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
            
            freqs = []
            for i, _ in enumerate(model.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            if base_spec_size != model_kwargs['img_size']:
                end_x, end_y = base_spec_size[0] // pf, base_spec_size[1] // pt
            else:
                end_x, end_y = F // pf, T // pt
            
            t_x, t_y = init_t_xy(end_x=end_x, end_y=end_y)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)

        else:
            self.compute_cis = partial(
                compute_axial_cis, 
                dim = model.embed_dim//model.num_heads, 
                theta = rope_theta
            )
            
            freqs_cis = self.compute_cis(
                end_x = F // pf,
                end_y = T // pt
            )
            self.freqs_cis = freqs_cis

        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False
        


    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_tokens', 'latent_pos_embed', 'freqs']

    def sample_orders(self, bsz, seq_len):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward(self, x, return_mask=False):

        # get tokens
        _, _, H, W = x.shape
        x = self.model.patch_embed(x)

        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_masking(x, orders).unsqueeze(-1)
            x = torch.where(mask.bool(), self.mask_token, x)
        else:
            mask = None 
        
        if not 'eva02' in self.model_name:
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)
        else:
            x, _ = self.model._pos_embed(x)

        if self.num_latent_tokens:
            # insert latent tokens
            z = self.latent_tokens.expand(x.size(0), -1, -1)
            x = torch.cat([x, z + self.latent_pos_embed], dim=1)
            
        # pre layer norm
        if not 'eva02' in self.model_name:
            x = self.model.norm_pre(x)
            
        if self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
        elif self.rope_mixed and self.use_rope:
            if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size[0], end_y = H // self.patch_size[1])
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.model.blocks):
                x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size[0], end_y = H // self.patch_size[1])
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.model.blocks):
                x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                
        # x = self.model.blocks(x)
        if not 'eva02' in self.model_name:
            x = self.model.norm(x)
        else:
            x = self.model.fc_norm(x)

        if self.num_latent_tokens:
            # get z tokens as out
            out = x[:, -self.num_latent_tokens:]
        else:
            # get img tokens as out
            out = x[:, self.num_prefix_tokens:]
        
        if return_mask:
            return out, mask
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, in_channels=1,
                 model_name = 'vit_small_patch14_dinov2.lvd142m',
                 model_kwargs = None,
                 pretrained = True,
                 tuning_method = 'lora',
                 tuning_kwargs = None,
                 num_latent_tokens=32,
                 to_audio='linear',
                 codebook_embed_dim=32,
                 rope_theta = 100.0,
                 rope_mixed = False,
                 use_rope = False,
                 use_ape = True,
                 cls_token = True,
                 base_spec_size=(128, 1000),
                 config=None,
                 ):
        super().__init__()

        self.config = config
        self.model_name = model_name

        # load model
    
        model = create_model(
            self.model_name,
            pretrained=pretrained,
            **model_kwargs
        )

        self.n_mels = model_kwargs['img_size'][0]
        self.max_time_frames = model_kwargs['img_size'][1]
        self.patch_size = model_kwargs['patch_size']
        pf, pt = self.patch_size

        self.embed_dim = model.embed_dim
        # get num of aud tokens
        self.num_aud_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        
        # tuning method
        if tuning_method == 'full':
            self.model = model
        elif tuning_method == 'lora':
            Lora_config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], r=8)
            self.model = peft.get_peft_model(model, Lora_config)
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # latent tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # to audio
        self.to_audio = ToAudio(to_audio=to_audio, spec_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_freq=pf, patch_time=pt)


        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta

        if self.rope_mixed and self.use_rope:
            self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
            
            freqs = []
            for i, _ in enumerate(model.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            if base_spec_size != model_kwargs['img_size']:
                end_x, end_y = base_spec_size[0] // pf, base_spec_size[1] // pt
            else:
                end_x, end_y = F // pf, T // pt
            
            t_x, t_y = init_t_xy(end_x=end_x, end_y=end_y)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)

        else:
            self.compute_cis = partial(
                compute_axial_cis, 
                dim=model.embed_dim//model.num_heads, 
                theta=rope_theta
            )
            
            freqs_cis = self.compute_cis(
                end_x = F // pf,
                end_y = T // pt
            )
            self.freqs_cis = freqs_cis
            
        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False


        if 'movq' in self.model_name:
            self.use_movq = True 
            self.model.norm = MoVQNorm(codebook_embed_dim, model.embed_dim)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.model.blocks:
                if isinstance(block, MoVQBlockv2):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            if isinstance(self.model.norm, MoVQNorm):
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].bias, 0)
        else:
            self.use_movq = False 
            

        self.cls_token = cls_token
        if not self.cls_token:
            self.model.cls_token = None
            self.num_prefix_tokens -= 1
            self.model.num_prefix_tokens -= 1
            
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed', 'freqs']

    @property
    def last_layer(self):
        return self.to_audio.get_last_layer()


    def forward(self, z, interpolate_zq=None, H=None, W=None):

        if H is None:
            num_aud_tokens = self.num_aud_tokens
            H = int(math.sqrt(num_aud_tokens)) * self.patch_size[0]
            W = int(math.sqrt(num_aud_tokens)) * self.patch_size[1]
        else:
            num_aud_tokens = H * W // (self.patch_size[0] * self.patch_size[1])

        # mask tokens
        if self.num_latent_tokens:
            if H is None:
                x = self.mask_token.expand(z.size(0), num_aud_tokens, -1)
            else:
                x = self.mask_token.expand(z.size(0), H * W // (self.patch_size[0] * self.patch_size[1]), -1)
        else:
            x = z 
            
        x = self.model._pos_embed(x, use_ape=self.use_ape)
        x = self.model.patch_drop(x)
        
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        x = self.model.norm_pre(x)
        
        
        if self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq=interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)
                
        elif self.rope_mixed and self.use_rope:
            if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)

        else:
            if self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq,  freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)      

        if self.use_movq:
            x = self.model.norm(x, interpolate_zq,  num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            x = self.model.norm(x)

        x = x[:, self.num_prefix_tokens:self.num_aud_tokens + self.num_prefix_tokens]

        out = self.to_audio(x)

        return out
