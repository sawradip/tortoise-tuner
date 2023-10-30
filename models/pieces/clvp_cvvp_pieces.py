import math
from functools import partial
from inspect import isfunction
from collections import namedtuple
from einops import rearrange, repeat

import torch
from torch import nn, einsum
import torch.nn.functional as F

from models.pieces.xtransformers import ContinuousTransformerWrapper, Encoder
from models.pieces.transformer import Transformer
from models.pieces.common_pieces import AttentionBlock

import utils.torch_intermediary as ml


def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]



class CheckpointedLayer(nn.Module):
    """
    Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
    checkpoint for all other args.
    """
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x, *args, **kwargs):
        for k, v in kwargs.items():
            assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
        partial = partial(self.wrap, **kwargs)
        return partial(x, *args)


class CheckpointedXTransformerEncoder(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, needs_permute=True, exit_permute=True, checkpoint=True, **xtransformer_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(**xtransformer_kwargs)
        self.needs_permute = needs_permute
        self.exit_permute = exit_permute

        if not checkpoint:
            return
        for i in range(len(self.transformer.attn_layers.layers)):
            n, b, r = self.transformer.attn_layers.layers[i]
            self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, x, **kwargs):
        if self.needs_permute:
            x = x.permute(0,2,1)
        h = self.transformer(x, **kwargs)
        if self.exit_permute:
            h = h.permute(0,2,1)
        return h

#### CVVP ####


# class AttentionBlock(nn.Module):
#     """
#     An attention block that allows spatial positions to attend to each other.

#     Originally ported from here, but adapted to the N-d case.
#     https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
#     """

#     def __init__(
#         self,
#         channels,
#         num_heads=1,
#         num_head_channels=-1,
#         do_checkpoint=True,
#         relative_pos_embeddings=False,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.do_checkpoint = do_checkpoint
#         if num_head_channels == -1:
#             self.num_heads = num_heads
#         else:
#             assert (
#                 channels % num_head_channels == 0
#             ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
#             self.num_heads = channels // num_head_channels
#         self.norm = normalization(channels)
#         self.qkv = nn.Conv1d(channels, channels * 3, 1)
#         # split heads before split qkv
#         self.attention = QKVAttentionLegacy(self.num_heads)

#         self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
#         if relative_pos_embeddings:
#             self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
#         else:
#             self.relative_pos_embeddings = None

#     def forward(self, x, mask=None):
#         b, c, *spatial = x.shape
#         x = x.reshape(b, c, -1)
#         qkv = self.qkv(self.norm(x))
#         h = self.attention(qkv, mask, self.relative_pos_embeddings)
#         h = self.proj_out(h)
#         return (x + h).reshape(b, c, *spatial)


class ConvFormatEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # nn.Embedding
        self.emb = ml.Embedding(*args, **kwargs)

    def forward(self, x):
        y = self.emb(x)
        return y.permute(0, 2, 1)

class CollapsingTransformer(nn.Module):
    def __init__(self, model_dim, output_dims, heads, dropout, depth, mask_percentage=0, **encoder_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(
            max_seq_len=-1,
            use_pos_emb=False,
            attn_layers=Encoder(
                dim=model_dim,
                depth=depth,
                heads=heads,
                ff_dropout=dropout,
                ff_mult=1,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                **encoder_kwargs,
            ))
        self.pre_combiner = nn.Sequential(nn.Conv1d(model_dim, output_dims, 1),
                                          AttentionBlock(
            output_dims, num_heads=heads, do_checkpoint=False),
            nn.Conv1d(output_dims, output_dims, 1))
        self.mask_percentage = mask_percentage

    def forward(self, x, **transformer_kwargs):
        h = self.transformer(x, **transformer_kwargs)
        h = h.permute(0, 2, 1)
        h = self.pre_combiner(h).permute(0, 2, 1)
        if self.training:
            mask = torch.rand_like(h.float()) > self.mask_percentage
        else:
            mask = torch.ones_like(h.float()).bool()
        return masked_mean(h, mask)
