# import torch
# from models import UnifiedVoice

# # gpt = UnifiedVoice(model_dim=256, heads=4, train_solo_embeddings=True, use_mel_codes_as_input=True, max_conditioning_inputs=4)
# # l = gpt(torch.randn(2, 3, 80, 800),
# #         torch.randint(high=120, size=(2,120)),
# #         torch.tensor([32, 120]),
# #         torch.randint(high=8192, size=(2,250)),
# #         torch.tensor([250*256,195*256]))
# # gpt.text_forward(torch.randn(2,80,800), torch.randint(high=50, size=(2,80)), torch.tensor([32, 80]))

# # dimensionality = {
# #                 "max_mel_tokens": 604,
# #                 "max_text_tokens": 402,
# #                 "max_conditioning_inputs": 2,
# #                 "layers": 30,
# #                 "model_dim": 1024,
# #                 "heads": 16,
# #                 "number_text_tokens": 255,
# #                 "start_text_token": 255,
# #                 "checkpointing": False,
# #                 "train_solo_embeddings": False
# #             }

# # autoregressive = UnifiedVoice(**dimensionality).cpu().eval()
# # autoregressive.load_state_dict(torch.load("/home/gpuserver/Desktop/sawradip/tortoise_tuner/ai-voice-cloning/models/tortoise/autoregressive.pth"))


# # from models import CLVP
# # clip = CLVP(text_mask_percentage=.2, voice_mask_percentage=.2)
# # clip(torch.randint(0,256,(2,120)),
# #         torch.tensor([50,100]),
# #         torch.randint(0,8192,(2,250)),
# #         torch.tensor([101,102]),
# #         return_loss=True)
# # nonloss = clip(torch.randint(0,256,(2,120)),
# #         torch.tensor([50,100]),
# #         torch.randint(0,8192,(2,250)),
# #         torch.tensor([101,102]),
# #         return_loss=False)
# # print(nonloss.shape)

# # clvp = CLVP(dim_text=768, dim_speech=768, dim_latent=768, num_text_tokens=256, text_enc_depth=20,
# #                          text_seq_len=350, text_heads=12,
# #                          num_speech_tokens=8192, speech_enc_depth=20, speech_heads=12, speech_seq_len=430,
# #                          use_xformers=True).cpu().eval()

# # clvp.load_state_dict(torch.load("/home/gpuserver/Desktop/sawradip/tortoise_tuner/ai-voice-cloning/models/tortoise/clvp2.pth"))
# # clvp(torch.randint(0,256,(2,120)),
# #         torch.tensor([50,100]),
# #         torch.randint(0,8192,(2,250)),
# #         torch.tensor([101,102]),
# #         return_loss=False)


# from models import CVVP

# # cvvp = CVVP(model_dim=512, 
# #             transformer_heads=8, 
# #             dropout=0, 
# #             mel_codes=8192, 
# #             conditioning_enc_depth=8, 
# #             cond_mask_percentage=0,
# #             speech_enc_depth=8, 
# #             speech_mask_percentage=0,
# #             latent_multiplier=1).cpu().eval()

# # cvvp.load_state_dict(torch.load("/home/gpuserver/Desktop/sawradip/tortoise_tuner/ai-voice-cloning/models/tortoise/cvvp.pth"))


# # from models import DiffusionTts

# # dimensionality = {
# #     "model_channels": 1024,
# #     "num_layers": 10,
# #     "in_channels": 100,
# #     "out_channels": 200,
# #     "in_latent_channels": 1024,
# #     "in_tokens": 8193,
# #     "dropout": 0,
# #     "use_fp16": False,
# #     "num_heads": 16,
# #     "layer_drop": 0,
# #     "unconditioned_percentage": 0
# # }
# # diffusion = DiffusionTts(**dimensionality)
# # diffusion.load_state_dict(torch.load("/home/gpuserver/Desktop/sawradip/tortoise_tuner/ai-voice-cloning/models/tortoise/diffusion_decoder.pth"))
        
# # from models import UnivNetGenerator
# # model = UnivNetGenerator()

# # c = torch.randn(3, 100, 10)
# # z = torch.randn(3, 64, 10)
# # print(c.shape)

# # y = model(c, z)
# # print(y.shape)
# # assert y.shape == torch.Size([3, 1, 2560])

# # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # print(pytorch_total_params)

# from models import BigVGAN

# vocoder_key = 'generator'

# vocoder = BigVGAN(config="/home/gpuserver/Desktop/sawradip/tortoise_tuner/ai-voice-cloning/models/tortoise/bigvgan_24khz_100band.json")
# vocoder.load_state_dict(torch.load("/home/gpuserver/Desktop/sawradip/tortoise_tuner/ai-voice-cloning/models/tortoise/bigvgan_24khz_100band.pth", map_location=torch.device('cpu'))[vocoder_key])
# c = torch.randn(3, 100, 10)
# z = torch.randn(3, 64, 10)
# print(c.shape)

# y = vocoder(c, z)
# print(y.shape)
# assert y.shape == torch.Size([3, 1, 2560])

# pytorch_total_params = sum(p.numel() for p in vocoder.parameters() if p.requires_grad)
# print(pytorch_total_params)

from models.tortoise import TortoiseModel


tts = TortoiseModel()

del tts