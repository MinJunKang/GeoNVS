
import math
from typing import List
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from diffusers.models.normalization import RMSNorm, LpNorm, LayerNorm
from diffusers.models.attention import FeedForward
from .embedding import position_grid_to_embed

# NOTE: `.rope` (requires transformer_engine) is imported lazily so that the
# dependency is only needed for the `rattn` / `gattn` fusion methods.


@torch.no_grad()
def normalize_3d_keypoints(kpts, pts_all):
    """ Normalize 3d keypoints locations based on the tight box
    kpts: [b, v, h, w, 3]
    """
    center = pts_all.mean(dim=1, keepdim=True) # B*1*3
    min_bound = pts_all.min(dim=1, keepdim=True).values # B*1*3
    max_bound = pts_all.max(dim=1, keepdim=True).values # B*1*3
    scale = torch.linalg.norm(max_bound - min_bound, dim=-1, keepdim=True) / 2.0 # B*1*1
    kpts_rescaled = (kpts - center[:, :, None, None, :]) / scale[:, :, None, None, :]
    return kpts_rescaled



class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.emb_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.emb_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.emb_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.emb_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.emb_q(x)
        context = context if context is not None else x
        k = self.emb_k(context)
        v = self.emb_v(context)
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
            (q, k, v),
        )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.emb_out(out)
        return out
    
    
def prepare_2d_rope_freqs_fused(h, w, dim, device, base=10000):
    """
    Prepare fused 2D axial RoPE frequencies for the
    transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb kernel.
    """
    dim_h = dim // 2
    dim_w = dim - dim_h

    def get_inv_freq(d):
        # step is 2 because channels are rotated in pairs
        return 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))

    inv_freq_h = get_inv_freq(dim_h) # (dim_h/2,)
    inv_freq_w = get_inv_freq(dim_w) # (dim_w/2,)

    t_h = torch.arange(h, device=device).float()
    t_w = torch.arange(w, device=device).float()

    freqs_h = torch.einsum("i,j->ij", t_h, inv_freq_h)  # (h, half_dim/2)
    freqs_w = torch.einsum("i,j->ij", t_w, inv_freq_w)  # (w, half_dim/2)

    freqs_h = freqs_h[:, None, :].expand(-1, w, -1)
    freqs_w = freqs_w[None, :, :].expand(h, -1, -1)

    # Combine axial 2D angles (h, w, dim/2). The TE kernel expects the angles
    # duplicated along the last dim, so we cat [angles, angles] to match `dim`.
    combined_freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # (h, w, dim/2)
    fused_freqs = rearrange(combined_freqs, "h w d -> (h w) 1 1 d") # (L, 1, 1, dim/2)
    fused_freqs = torch.cat([fused_freqs, fused_freqs], dim=-1) # (L, 1, 1, dim/2*2) -> (L, 1, 1, dim)

    return fused_freqs
    
    
class RoPEAttention(nn.Module):
    def __init__(self, query_dim, h, w, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.emb_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.emb_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.emb_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.emb_out = nn.Linear(inner_dim, query_dim)

        self.register_buffer('rope_freqs', prepare_2d_rope_freqs_fused(h, w, dim_head, 'cpu'), persistent=False)

    def forward(self, x, context):
        from .rope import apply_rotary_pos_emb

        # x, context: (b, l, c)
        q = self.emb_q(x)
        k = self.emb_k(context)
        v = self.emb_v(context)

        # split heads: (b, l, h, d)
        q = rearrange(q, "b l (h d) -> l b h d", h=self.heads)
        k = rearrange(k, "b l (h d) -> l b h d", h=self.heads)
        v = rearrange(v, "b l (h d) -> l b h d", h=self.heads)

        # apply Transformer Engine's fused RoPE
        # freqs has shape [L, 1, 1, D/2] and is rotated like a complex product
        q = apply_rotary_pos_emb(q, self.rope_freqs.to(q.dtype), tensor_format="sbhd", fused=True)
        k = apply_rotary_pos_emb(k, self.rope_freqs.to(k.dtype), tensor_format="sbhd", fused=True)

        # 4. SDPA (B H L D format)
        q, k, v = map(lambda t: rearrange(t, "l b h d -> b h l d"), (q, k, v))

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, "b h l d -> b l (h d)")
        return self.emb_out(out)



class FusionLayer(nn.Module):
    
    def __init__(
        self, 
        channel_dim: int, 
        h: int,
        w: int,
        fusion_method: str, 
        zero_init: bool = True
    ):
        super().__init__()
        assert fusion_method in ['sum', 'concat', 'attn', 'rattn', 'gattn']
        self.zero_init = zero_init
        self.fusion_method = fusion_method
        if fusion_method in ['sum', 'concat']:
            if fusion_method == 'sum':
                input_dim = channel_dim
            elif fusion_method == 'concat':
                input_dim = channel_dim * 2
            self.fusion_layer = FeedForward(
                input_dim,
                dim_out=channel_dim,
                activation_fn="geglu"
            )
        elif fusion_method == 'attn':
            num_heads = 8
            input_dim = channel_dim
            self.gate = nn.Parameter(torch.zeros(1))
            self.fusion_layer = Attention(
                query_dim=input_dim,
                context_dim=input_dim,
                heads=num_heads,
                dim_head=input_dim
            )
        elif fusion_method == 'rattn':
            num_heads = 4
            # maybe use spiral RoPE in the future from https://github.com/huajianduzhuo-code/Spiral_RoPE
            input_dim = channel_dim
            self.fusion_layer = RoPEAttention(
                query_dim=input_dim,
                context_dim=input_dim,
                h=h, w=w,
                heads=num_heads,
                dim_head=input_dim
            )
        elif fusion_method == 'gattn':
            num_heads = 4
            input_dim = channel_dim
            self.gate_net = nn.Sequential(
                nn.Linear(channel_dim * 2, channel_dim),
                nn.SiLU(),
                nn.Linear(channel_dim, 1),
                nn.Tanh()
            )
            self.fusion_layer = RoPEAttention(
                query_dim=input_dim,
                context_dim=input_dim,
                h=h, w=w,
                heads=num_heads,
                dim_head=input_dim
            )
        self.pre_norm = LayerNorm(input_dim, elementwise_affine=False)
        self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if self.zero_init:
                nn.init.constant_(m.weight, 0)
            else:
                trunc_normal_(m.weight, std=.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

        # gate_net initialization
        if self.fusion_method == 'gattn' and hasattr(self, 'gate_net'):
            last_linear = self.gate_net[-2] 
            nn.init.constant_(last_linear.weight, 0)
            nn.init.constant_(last_linear.bias, 0)
        
    def forward(self, feat_src, feat_tar):
        feat_src = feat_src.permute(0, 2, 3, 1)  # B H W C
        feat_tar = feat_tar.permute(0, 2, 3, 1)  # B H W C
        
        if self.fusion_method == "concat":
            feat_inputs = torch.cat([feat_src, feat_tar], dim=-1)
            feat_merged = self.fusion_layer(self.pre_norm(feat_inputs))
        elif self.fusion_method == "sum":
            feat_inputs = feat_src + feat_tar
            feat_merged = self.fusion_layer(self.pre_norm(feat_inputs))
        elif self.fusion_method == 'attn':
            hs, ws = feat_src.shape[1], feat_src.shape[2]
            query = rearrange(self.pre_norm(feat_src), 'b h w c -> b (h w) c')
            context = rearrange(self.pre_norm(feat_tar), 'b h w c -> b (h w) c')
            feat_merged = rearrange(self.fusion_layer(query, context=context), 'b (h w) c -> b h w c', h=hs, w=ws)
            feat_merged = feat_src + self.gate.to(feat_merged.dtype) * feat_merged
        elif self.fusion_method == 'rattn':
            hs, ws = feat_src.shape[1], feat_src.shape[2]
            query = rearrange(self.pre_norm(feat_src), 'b h w c -> b (h w) c')
            context = rearrange(self.pre_norm(feat_tar), 'b h w c -> b (h w) c')
            feat_merged = rearrange(self.fusion_layer(query, context=context), 'b (h w) c -> b h w c', h=hs, w=ws)
            feat_merged = feat_src + feat_merged
        elif self.fusion_method == 'gattn':
            hs, ws = feat_src.shape[1], feat_src.shape[2]
            query = rearrange(self.pre_norm(feat_src), 'b h w c -> b (h w) c')
            context = rearrange(self.pre_norm(feat_tar), 'b h w c -> b (h w) c')
            feat_merged = rearrange(self.fusion_layer(query, context=context), 'b (h w) c -> b h w c', h=hs, w=ws)
            gate_input = torch.cat([feat_src, feat_merged], dim=-1)
            feat_merged = feat_src + self.gate_net(gate_input) * feat_merged
        else:
            raise NotImplementedError(f"Unknown fusion method: {self.fusion_method}")
        
        feat_merged = feat_merged.permute(0, 3, 1, 2)  # B H W C -> B C H W
        return feat_merged
    
    
class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k//2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)
    
    
class ResNetConvs(nn.Module):
    def __init__(
        self, 
        channel, 
        hidden_feature_dim=128, 
        use_pos_emb=True, 
        render_color=False,
        render_uncertainty=False, 
        eps=1e-6
    ):
        super(ResNetConvs, self).__init__()
        self.use_pos_emb = use_pos_emb
        self.render_uncertainty = render_uncertainty
        input_channel = channel + int(render_uncertainty) + int(render_color) * 3
        self.conv1 = resconv(input_channel, hidden_feature_dim, k=3, s=1)
        self.conv2 = resconv(hidden_feature_dim, hidden_feature_dim, k=3, s=1)
        self.proj = nn.Conv2d(hidden_feature_dim, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_layer = LayerNorm(channel, eps=eps, elementwise_affine=False)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feature, color=None, uc=None, gs_pe_norm=None):
        x = feature
        if gs_pe_norm is not None and self.use_pos_emb:
            pe = position_grid_to_embed(gs_pe_norm.permute(0, 2, 3, 1), feature.shape[1])
            x = x + pe.permute(0, 3, 1, 2).to(x.dtype)
        if color is not None and self.render_color:
            x = torch.cat([x, color], dim=1)
        if uc is not None and self.render_uncertainty:
            x = torch.cat([x, uc], dim=1)
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out = self.proj(out_2) + feature  # residual connection
        out_norm = self.norm_layer(out.permute(0, 2, 3, 1))  # input to attention layer
        return out, out_norm


class GSAdapter(nn.Module):
    
    def __init__(
        self,
        gs_decoder,
        gs_channel_dim: int,
        detach_input: bool = True,
        recover_net_cfg: dict = {},
        use_gs_positional_encoding: bool = True,
        gs_rotation_encoding: bool = False,
        use_refinenet: bool = True,
        render_color: bool = False,
        render_uncertainty: bool = False,
        use_feature_loss: bool = False,
        feature_loss_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.gs_channel_dim = gs_channel_dim
        self.gs_decoder = gs_decoder
        self.detach_input = detach_input
        self.render_color = render_color
        self.render_uncertainty = render_uncertainty
        self.use_refinenet = use_refinenet
        self.use_feature_loss = use_feature_loss
        self.feature_loss_weight = feature_loss_weight
        self.use_gs_positional_encoding = use_gs_positional_encoding
        self.gs_rotation_encoding = gs_rotation_encoding
        self.eps = float(eps)
        
        # recovering low-freq rendered feature to high-freq feature space
        if self.use_refinenet:
            self.recover_net = ResNetConvs(
                channel=gs_channel_dim, 
                render_uncertainty=render_uncertainty, 
                render_color=render_color,
                use_pos_emb=use_gs_positional_encoding, 
                eps=self.eps, 
                **recover_net_cfg
            )
        
    def forward(
        self, 
        feature, 
        gs_batch, 
        input_mask,
        scale_invariant: bool,
        e3nn: bool = True,
    ):
        input_dtype = feature.dtype
        iv_feature, nv_feature = feature[input_mask], feature[~input_mask]
        if self.detach_input:
            iv_feature = iv_feature.detach()
        
        num_gaussian = gs_batch['gaussian_param'].means.shape[1]
        batch_size, num_frames, height, width, knum = gs_batch['index'].shape

        # Static index preparation for the pixel-to-Gaussian scatter. Everything
        # here depends only on gs_batch and input_mask, so when gs_batch is
        # reused across denoising steps (see the sampling-time cache in the
        # backbones) it is computed once per scene.
        cache = gs_batch.get('_iv_scatter_cache')
        if cache is not None and cache[0] is input_mask:
            _, gsindex_iv, gsweight_iv, pixel_index = cache
        else:
            bsz_indices = repeat(torch.arange(batch_size, device=gs_batch['index'].device), 'b -> b v', v=num_frames)
            input_mask_ = rearrange(input_mask, '(b v) -> b v', v=num_frames)
            gsindex_iv, gsweight_iv = gs_batch['index'][input_mask_], gs_batch['weight'][input_mask_]
            valid_mask = gsindex_iv >= 0
            gsindex_iv = bsz_indices[input_mask_][:, None, None, None] * num_gaussian + gsindex_iv.clamp(min=0)
            gsindex_iv, gsweight_iv = gsindex_iv[valid_mask], gsweight_iv[valid_mask]
            # flat (view, h, w) pixel id of every valid entry: the feature of a
            # pixel is shared by all its k entries, so per-pixel features can be
            # gathered directly instead of materializing a knum-times repeated
            # feature tensor before masking
            num_iv = valid_mask.shape[0]
            pixel_ids = torch.arange(num_iv * height * width, device=valid_mask.device)
            pixel_ids = pixel_ids.view(num_iv, height, width, 1).expand(-1, -1, -1, knum)
            pixel_index = pixel_ids[valid_mask]
            gs_batch['_iv_scatter_cache'] = (input_mask, gsindex_iv, gsweight_iv, pixel_index)

        # direct embedding of input-view 2D features to gaussian splats
        emb_features = rearrange(iv_feature, 'n c h w -> (n h w) c')[pixel_index]
        feature_buffer = torch.zeros(batch_size * num_gaussian, self.gs_channel_dim, device=emb_features.device)
        weight_buffer = torch.zeros(batch_size * num_gaussian, 1, device=gsweight_iv.device)
        feature_buffer.scatter_add_(0, gsindex_iv[:, None].expand(-1, self.gs_channel_dim), emb_features * gsweight_iv[:, None])
        weight_buffer.scatter_add_(0, gsindex_iv[:, None], gsweight_iv[:, None])
        feature_buffer = rearrange(feature_buffer / (weight_buffer + self.eps), '(b n) c -> b n c', b=batch_size)
        
        # normalize feature buffer
        feature_buffer = F.normalize(feature_buffer, dim=-1, eps=self.eps)

        # rasterize to novel viewpoint
        output = self.gs_decoder(
            gaussians=gs_batch['gaussian_param'],
            gaussian_features=feature_buffer.float(),
            extrinsics=gs_batch['extrinsics'],
            intrinsics=gs_batch['intrinsics'],
            near=gs_batch['near'],
            far=gs_batch['far'],
            image_shape=(height, width),
            scale_invariant=scale_invariant,
            render_color=self.render_color,
            render_feature=True,
            e3nn=e3nn,
            render_uncertainty=self.render_uncertainty,
        )
        
        # feature restoration
        rendered_feature = rearrange(output.feature.to(input_dtype), 'b t c h w -> (b t) c h w')
        rendered_uc = rearrange(output.uncertainty.to(input_dtype), 'b t h w -> (b t) 1 h w') if self.render_uncertainty else None
        rendered_color = rearrange(output.color.to(input_dtype), 'b t c h w -> (b t) c h w') if self.render_color else None
        
        # gaussian positional encoding
        with torch.no_grad():
            if self.use_gs_positional_encoding and self.use_refinenet and 'gspe' in gs_batch:
                # reuse the encoding computed on a previous denoising step
                # (gs_batch is cached across the sampling loop and the encoding
                # only depends on its static entries)
                gspe_sampled = gs_batch['gspe']
            elif self.use_gs_positional_encoding and self.use_refinenet:
                # weight is already sorted in descending order
                max_weightv, max_gsindex = gs_batch['weight'][..., 0], gs_batch['index'][..., 0]
                gspts_weight = rearrange(max_weightv, 'b t h w -> (b t) 1 h w')
                valid_mask_gsindex = max_gsindex > 0
                
                # sampling 3D points from gaussian centers
                gspts = gs_batch['gaussian_param'].means[:, None, None, None]  # (B,1,1,1,N,3)
                gspts = gspts.expand(batch_size, *max_gsindex.shape[1:], num_gaussian, 3)  # (B,T,H,W,N,3)
                gidx = max_gsindex[..., None, None].expand(batch_size, *max_gsindex.shape[1:], 1, 3)  # (B,T,H,W,1,3)
                gspts_sampled = rearrange(torch.gather(gspts, dim=4, index=gidx.clamp(min=0)), 'b t h w 1 c -> b t h w c')
                gspts_sampled_norm = normalize_3d_keypoints(gspts_sampled, gs_batch['gaussian_param'].means)
                gspts_sampled_norm[~valid_mask_gsindex] = -1.0  # give boundary value to invalid index
                gspts_sampled = rearrange(gspts_sampled_norm, 'b t h w c -> (b t) c h w')
                
                if self.gs_rotation_encoding:
                    gsrot = gs_batch['gaussian_param'].rotations[:, None, None, None]  # (B,1,1,1,N,4)
                    gsrot = gsrot.expand(batch_size, *max_gsindex.shape[1:], num_gaussian, 4)  # (B,T,H,W,N,4)
                    gidx = max_gsindex[..., None, None].expand(batch_size, *max_gsindex.shape[1:], 1, 4)  # (B,T,H,W,1,4)
                    gsrot_sampled = rearrange(torch.gather(gsrot, dim=4, index=gidx.clamp(min=0)), 'b t h w 1 c -> b t h w c')
                    gsrot_sampled[~valid_mask_gsindex][:, 0] = 1.0  # give boundary value to invalid index
                    gsrot_sampled[~valid_mask_gsindex][:, 1:] = 0.0
                    gsrot_sampled = rearrange(gsrot_sampled, 'b t h w c -> (b t) c h w')
                    gspe_sampled = torch.cat([gspts_sampled, gsrot_sampled, gspts_weight], dim=1).to(input_dtype)
                else:
                    gspe_sampled = torch.cat([gspts_sampled, gspts_weight], dim=1).to(input_dtype)
                gs_batch['gspe'] = gspe_sampled
            else:
                gspe_sampled = None
        
        # recovering feature
        if self.use_refinenet:
            feat_aug, _ = self.recover_net(rendered_feature, uc=rendered_uc, gs_pe_norm=gspe_sampled, color=rendered_color)
        else:
            feat_aug = rendered_feature
        
        # novel-view features
        updated_features = feat_aug[~input_mask]

        ### Feature Constraint Loss ###
        if self.use_feature_loss and self.use_refinenet:
            # Cosine similarity loss
            feature_rec_loss = self.feature_loss_weight * ((1 - F.cosine_similarity(feat_aug[input_mask], iv_feature, dim=1)) * 0.5).mean()
        else:
            feature_rec_loss = 0.0
        
        return updated_features, feature_rec_loss
    
    
class GSDecoder(nn.Module):
    
    def __init__(
        self, 
        gs_channel_dim: int,
        feature_dims: List[int],
    ):
        super(GSDecoder, self).__init__()
        self.gs_channel_dim = gs_channel_dim
        self.feature_dims = feature_dims
        # projection layers
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=gs_channel_dim, 
                out_channels=dim,
                kernel_size=2**i,
                stride=2**i,
                padding=0
            )
            for i, dim in enumerate(feature_dims)
        ])
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, encoder_features):
        # Project each feature map to align with diffusion features
        output_features = deque()
        for idx, proj in enumerate(self.proj_layers):
            output_features.append(proj(x) + encoder_features[idx])
        return output_features
