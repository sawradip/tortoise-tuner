import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

import utils.torch_intermediary as ml
from utils.tools import ACC_DEVICE, CPU_DEVICE
from models.pieces.diffusion_pieces import is_latent, timestep_embedding, DiffusionLayer, TimestepEmbedSequential, ResBlock
from models.pieces.common_pieces import normalization, AttentionBlock

class DiffusionTts(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=100,
            in_latent_channels=512,
            in_tokens=8193,
            out_channels=200,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=16,
            # Parameters for regularization.
            layer_drop=.1,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop

        self.inp_block = nn.Conv1d(in_channels, model_channels, 3, 1, 1)
        self.time_embed = nn.Sequential(
            # nn.Linear
            ml.Linear(model_channels, model_channels),
            nn.SiLU(),
            # nn.Linear
            ml.Linear(model_channels, model_channels),
        )

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        # nn.Embedding
        self.code_embedding = ml.Embedding(in_tokens, model_channels)
        self.code_converter = nn.Sequential(
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.code_norm = normalization(model_channels)
        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_latent_channels, model_channels, 3, padding=1),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels,model_channels,3,padding=1,stride=2),
                                                 nn.Conv1d(model_channels, model_channels*2,3,padding=1,stride=2),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False))
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
        )

        self.integrating_conv = nn.Conv1d(model_channels*2, model_channels, kernel_size=1)
        self.mel_head = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)] +
                                    [ResBlock(model_channels, model_channels, dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, 3, padding=1),
        )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'minicoder': list(self.contextual_embedder.parameters()),
            'layers': list(self.layers.parameters()),
            'code_converters': list(self.code_embedding.parameters()) + list(self.code_converter.parameters()) + list(self.latent_conditioner.parameters()) + list(self.latent_conditioner.parameters()),
            'timestep_integrator': list(self.conditioning_timestep_integrator.parameters()) + list(self.integrating_conv.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups
    
        
    def get_conditioning(self, conditioning_input):
        speech_conditioning_input = conditioning_input.unsqueeze(1) if len(
            conditioning_input.shape) == 3 else conditioning_input
        conds = []
        print('Obtaining Dffusion Conditioning')
        for j in range(speech_conditioning_input.shape[1]):
            conditioning_input_piece = speech_conditioning_input[:, j].to(ACC_DEVICE)
            conditioning_output_piece = self.contextual_embedder(conditioning_input_piece)
            conditioning_output_piece_cpu = conditioning_output_piece.to(CPU_DEVICE)
            del conditioning_output_piece
            conds.append(conditioning_output_piece_cpu)
        conds = torch.cat(conds, dim=-1)
        conds = conds.mean(dim=-1)
        return conds

    def timestep_independent(self, aligned_conditioning, conditioning_latent, expected_seq_len, return_code_pred):
        # Shuffle aligned_latent to BxCxS format
        if is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        cond_scale, cond_shift = torch.chunk(conditioning_latent, 2, dim=1)
        if is_latent(aligned_conditioning):
            code_emb = self.latent_conditioner(aligned_conditioning)
        else:
            code_emb = self.code_embedding(aligned_conditioning).permute(0, 2, 1)
            code_emb = self.code_converter(code_emb)
        code_emb = self.code_norm(code_emb) * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)

        unconditioned_batches = torch.zeros((code_emb.shape[0], 1, 1), device=code_emb.device)
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(aligned_conditioning.shape[0], 1, 1),
                                   code_emb)
        expanded_code_emb = F.interpolate(code_emb, size=expected_seq_len, mode='nearest')

        if not return_code_pred:
            return expanded_code_emb
        else:
            mel_pred = self.mel_head(expanded_code_emb)
            # Multiply mel_pred by !unconditioned_branches, which drops the gradient on unconditioned branches. This is because we don't want that gradient being used to train parameters through the codes_embedder as it unbalances contributions to that network from the MSE loss.
            mel_pred = mel_pred * unconditioned_batches.logical_not()
            return expanded_code_emb, mel_pred

    def forward(self, x, timesteps, aligned_conditioning=None, conditioning_latent=None, precomputed_aligned_embeddings=None, conditioning_free=False, return_code_pred=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param aligned_conditioning: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_latent: a pre-computed conditioning latent; see get_conditioning().
        :param precomputed_aligned_embeddings: Embeddings returned from self.timestep_independent()
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert precomputed_aligned_embeddings is not None or (aligned_conditioning is not None and conditioning_latent is not None)
        assert not (return_code_pred and precomputed_aligned_embeddings is not None)  # These two are mutually exclusive.

        unused_params = []
        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
            unused_params.extend(list(self.code_converter.parameters()) + list(self.code_embedding.parameters()))
            unused_params.extend(list(self.latent_conditioner.parameters()))
        else:
            if precomputed_aligned_embeddings is not None:
                code_emb = precomputed_aligned_embeddings
            else:
                code_emb, mel_pred = self.timestep_independent(aligned_conditioning, conditioning_latent, x.shape[-1], True)
                if is_latent(aligned_conditioning):
                    unused_params.extend(list(self.code_converter.parameters()) + list(self.code_embedding.parameters()))
                else:
                    unused_params.extend(list(self.latent_conditioner.parameters()))

            unused_params.append(self.unconditioned_embedding)

        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        code_emb = self.conditioning_timestep_integrator(code_emb, time_emb)
        x = self.inp_block(x)
        x = torch.cat([x, code_emb], dim=1)
        x = self.integrating_conv(x)
        for i, lyr in enumerate(self.layers):
            # Do layer drop where applicable. Do not drop first and last layers.
            if self.training and self.layer_drop > 0 and i != 0 and i != (len(self.layers)-1) and random.random() < self.layer_drop:
                unused_params.extend(list(lyr.parameters()))
            else:
                # First and last blocks will have autocast disabled for improved precision.
                # x.device.type
                with autocast(device_type='cuda', enabled=self.enable_fp16 and i != 0):
                    x = lyr(x, time_emb)

        x = x.float()
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        if return_code_pred:
            return out, mel_pred
        return out