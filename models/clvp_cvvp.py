import torch
from torch import nn, einsum
import torch.nn.functional as F

import utils.torch_intermediary as ml
# from models.pieces.xtransformers import Encoder
from models.pieces.clvp_cvvp_pieces import masked_mean, CheckpointedXTransformerEncoder, CollapsingTransformer, ConvFormatEmbedding, Encoder, Transformer
from models.pieces.common_pieces import do_gc

class CLVP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between tokenized audio data and the corresponding
    transcribed text.

    Originally from https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """

    def __init__(
            self,
            *,
            dim_text=512,
            dim_speech=512,
            dim_latent=512,
            num_text_tokens=256,
            text_enc_depth=6,
            text_seq_len=120,
            text_heads=8,
            num_speech_tokens=8192,
            speech_enc_depth=6,
            speech_heads=8,
            speech_seq_len=250,
            text_mask_percentage=0,
            voice_mask_percentage=0,
            wav_token_compression=1024,
            use_xformers=False,
    ):
        super().__init__()
        # nn.Embedding
        self.text_emb = ml.Embedding(num_text_tokens, dim_text)
        # nn.Linear
        self.to_text_latent = ml.Linear(dim_text, dim_latent, bias=False)

        # nn.Embedding
        self.speech_emb = ml.Embedding(num_speech_tokens, dim_speech)
        # nn.Linear
        self.to_speech_latent = ml.Linear(dim_speech, dim_latent, bias=False)

        if use_xformers:
            self.text_transformer = CheckpointedXTransformerEncoder(
                needs_permute=False,
                exit_permute=False,
                max_seq_len=-1,
                attn_layers=Encoder(
                    dim=dim_text,
                    depth=text_enc_depth,
                    heads=text_heads,
                    ff_dropout=.1,
                    ff_mult=2,
                    attn_dropout=.1,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                ))
            self.speech_transformer = CheckpointedXTransformerEncoder(
                needs_permute=False,
                exit_permute=False,
                max_seq_len=-1,
                attn_layers=Encoder(
                    dim=dim_speech,
                    depth=speech_enc_depth,
                    heads=speech_heads,
                    ff_dropout=.1,
                    ff_mult=2,
                    attn_dropout=.1,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                ))
        else:
            self.text_transformer = Transformer(causal=False, seq_len=text_seq_len, dim=dim_text, depth=text_enc_depth,
                                                heads=text_heads)
            self.speech_transformer = Transformer(causal=False, seq_len=speech_seq_len, dim=dim_speech,
                                                  depth=speech_enc_depth, heads=speech_heads)

        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_mask_percentage = text_mask_percentage
        self.voice_mask_percentage = voice_mask_percentage
        self.wav_token_compression = wav_token_compression
        self.xformers = use_xformers
        if not use_xformers:
            # nn.Embedding
            self.text_pos_emb = ml.Embedding(text_seq_len, dim_text)
            # nn.Embedding
            self.speech_pos_emb = ml.Embedding(num_speech_tokens, dim_speech)

    def forward(
            self,
            text,
            speech_tokens,
            return_loss=False
    ):
        b, device = text.shape[0], text.device
        if self.training:
            text_mask = torch.rand_like(text.float()) > self.text_mask_percentage
            voice_mask = torch.rand_like(speech_tokens.float()) > self.voice_mask_percentage
        else:
            text_mask = torch.ones_like(text.float()).bool()
            voice_mask = torch.ones_like(speech_tokens.float()).bool()

        text_emb = self.text_emb(text)
        speech_emb = self.speech_emb(speech_tokens)

        if not self.xformers:
            text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))
            speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=device))

        
        text_latents = self.to_text_latent(masked_mean(self.text_transformer(text_emb, mask=text_mask), text_mask, dim=1))

        # on ROCm at least, allocated VRAM spikes here
        do_gc()
        speech_latents = self.to_speech_latent(masked_mean(self.speech_transformer(speech_emb, mask=voice_mask), voice_mask, dim=1))
        do_gc()

        text_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, speech_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, speech_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, speech_latents) * temp
        labels = torch.arange(b, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


class CVVP(nn.Module):
    def __init__(
            self,
            model_dim=512,
            transformer_heads=8,
            dropout=.1,
            conditioning_enc_depth=8,
            cond_mask_percentage=0,
            mel_channels=80,
            mel_codes=None,
            speech_enc_depth=8,
            speech_mask_percentage=0,
            latent_multiplier=1,
    ):
        super().__init__()
        latent_dim = latent_multiplier*model_dim
        self.temperature = nn.Parameter(torch.tensor(1.))

        self.cond_emb = nn.Sequential(nn.Conv1d(mel_channels, model_dim//2, kernel_size=5, stride=2, padding=2),
                                      nn.Conv1d(model_dim//2, model_dim, kernel_size=3, stride=2, padding=1))
        self.conditioning_transformer = CollapsingTransformer(
            model_dim, model_dim, transformer_heads, dropout, conditioning_enc_depth, cond_mask_percentage)
        # nn.Linear
        self.to_conditioning_latent = ml.Linear(
            latent_dim, latent_dim, bias=False)

        if mel_codes is None:
            self.speech_emb = nn.Conv1d(
                mel_channels, model_dim, kernel_size=5, padding=2)
        else:
            self.speech_emb = ConvFormatEmbedding(mel_codes, model_dim)
        self.speech_transformer = CollapsingTransformer(
            model_dim, latent_dim, transformer_heads, dropout, speech_enc_depth, speech_mask_percentage)
        # nn.Linear
        self.to_speech_latent = ml.Linear(
            latent_dim, latent_dim, bias=False)

    def get_grad_norm_parameter_groups(self):
        return {
            'conditioning': list(self.conditioning_transformer.parameters()),
            'speech': list(self.speech_transformer.parameters()),
        }

    def forward(
            self,
            mel_cond,
            mel_input,
            return_loss=False
    ):
        cond_emb = self.cond_emb(mel_cond).permute(0, 2, 1)
        enc_cond = self.conditioning_transformer(cond_emb)
        cond_latents = self.to_conditioning_latent(enc_cond)

        speech_emb = self.speech_emb(mel_input).permute(0, 2, 1)
        enc_speech = self.speech_transformer(speech_emb)
        speech_latents = self.to_speech_latent(enc_speech)

        cond_latents, speech_latents = map(lambda t: F.normalize(
            t, p=2, dim=-1), (cond_latents, speech_latents))
        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', cond_latents,
                         speech_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', cond_latents,
                     speech_latents) * temp
        labels = torch.arange(
            cond_latents.shape[0], device=mel_input.device)
        loss = (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.t(), labels)) / 2

        return loss


if __name__ == '__main__':
    clip = CLVP(text_mask_percentage=.2, voice_mask_percentage=.2)
    clip(torch.randint(0,256,(2,120)),
         torch.tensor([50,100]),
         torch.randint(0,8192,(2,250)),
         torch.tensor([101,102]),
         return_loss=True)
    nonloss = clip(torch.randint(0,256,(2,120)),
         torch.tensor([50,100]),
         torch.randint(0,8192,(2,250)),
         torch.tensor([101,102]),
         return_loss=False)
    print(nonloss.shape)

    clvp = CVVP()
    clvp(torch.randn(2, 80, 100),
         torch.randn(2, 80, 95),
         return_loss=True)