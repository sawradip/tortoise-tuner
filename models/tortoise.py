import torch

from models.autoregressive import UnifiedVoice
from models.clvp_cvvp import CVVP, CLVP
from models.diffusion import DiffusionTts
from models.vocoder import BigVGAN, UnivNetGenerator

from utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from utils.tokenizer import VoiceBpeTokenizer
from utils.aligner import Wav2VecAlignment
from utils.preset_params import WEIGHT_PATHS


class TortoiseModel:
    def __init__(self,
                 enable_redaction = True,
                 enable_cvvp = False,
                 enable_optimization = 0, 
                 vocoder_name = 'bigvgan',
                 pretrained = False):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.enable_redaction = enable_redaction
        self.enable_cvvp = enable_cvvp
        self.enable_optimization = enable_optimization

        if self.enable_redaction:
            self.aligner = Wav2VecAlignment()


    def load_tokenizer(self):
        return VoiceBpeTokenizer(vocab_file=WEIGHT_PATHS['tokenizer_config'])
    
    def load_autoregressive(self):

        dimensionality = {
                "max_mel_tokens": 604,
                "max_text_tokens": 402,
                "max_conditioning_inputs": 2,
                "layers": 30,
                "model_dim": 1024,
                "heads": 16,
                "number_text_tokens": 255,
                "start_text_token": 255,
                "checkpointing": False,
                "train_solo_embeddings": False
            }

        autoregressive = UnifiedVoice(**dimensionality).eval()
        autoregressive.load_state_dict(torch.load(WEIGHT_PATHS['autoregressive']))
        autoregressive.post_init_gpt2_config(use_deepspeed=self.enable_optimization,
                                                kv_cache=self.enable_optimization)
        # self.autoregressive = self.autoregressive.to(self.device) 
        print("Loaded autoregrassive model")
        return autoregressive

    def load_weights(self):
        clvp = CLVP(dim_text=768, 
                         dim_speech=768, 
                         dim_latent=768, 
                         num_text_tokens=256, 
                         text_enc_depth=20,
                         text_seq_len=350, 
                         text_heads=12,
                         num_speech_tokens=8192, 
                         speech_enc_depth=20, 
                         speech_heads=12, 
                         speech_seq_len=430,
                         use_xformers=True).eval()

        clvp.load_state_dict(torch.load(WEIGHT_PATHS['clvp']))
        # self.clvp = self.clvp.to(self.device)
        print("Loaded clvp model")
        return clvp 

    def load_cvvp(self):
        cvvp = CVVP(model_dim=512, transformer_heads=8, dropout=0, mel_codes=8192, conditioning_enc_depth=8, cond_mask_percentage=0,
                            speech_enc_depth=8, speech_mask_percentage=0, latent_multiplier=1).cpu().eval()

        cvvp.load_state_dict(torch.load(WEIGHT_PATHS['cvvp']))
        # self.cvvp = self.cvvp.to(self.device)
        print("Loaded cvvp model")

        return cvvp


    def load_diffusion(self):
        dimensionality = {
            "model_channels": 1024,
            "num_layers": 10,
            "in_channels": 100,
            "out_channels": 200,
            "in_latent_channels": 1024,
            "in_tokens": 8193,
            "dropout": 0,
            "use_fp16": False,
            "num_heads": 16,
            "layer_drop": 0,
            "unconditioned_percentage": 0
        }
        diffusion = DiffusionTts(**dimensionality)
        diffusion.load_state_dict(torch.load(WEIGHT_PATHS['diffusion']))
        # self.diffusion = self.diffusion.to(self.device)
        print("Loaded diffusion model")
        return diffusion

    def load_vocoder(self, vocoder_name = 'bigvgan'):
        if vocoder_name == 'bigvgan':
            vocoder_key = 'generator'
            # if vocoder_config is None or not vocoder_config.endswith('json'):
            #     raise ValueError("vocoder_config must be provided for BigvGAN Vocoder")
            vocoder = BigVGAN(config=WEIGHT_PATHS['vocoder_bigvgan_config'])
        else:
            vocoder_key = 'model_g'
            vocoder = UnivNetGenerator()

        vocoder_weight = WEIGHT_PATHS['vocoder_bigvgan'] if self.vocoder_name=='bigvgan' else WEIGHT_PATHS['vocoder_univnet']  
        vocoder.load_state_dict(torch.load(vocoder_weight)[self.vocoder_key])
        # self.vocoder = self.vocoder.to(self.device)
        print("Loaded vocoder model")

        return vocoder
# if __name__ == '__main__':
#     tortoise_tts = TortoiseModel()







        



