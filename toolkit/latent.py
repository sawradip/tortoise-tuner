import os
import random

import torch
import torchaudio
import torch.nn.functional as F

from glob import glob
from tqdm.auto import tqdm
from models.tortoise import TortoiseModel

from utils.tools import ACC_DEVICE, CPU_DEVICE, do_gc
from utils.audio import load_audio, pad_or_truncate, wav_to_univnet_mel
from utils.helpers.mel import TorchMelSpectrogram

class TortoiseLatent():
    def __init__(self):
        super().__init__()
        self.input_sample_rate = 22050
        self.tortoise_model = TortoiseModel()
        self.autoregressive = self.tortoise_model.load_autoregressive()
        self.diffusion = self.tortoise_model.load_diffusion()
        # self.voice_sample = self.load_voice_files(audio_files)

    def load_voice_files(self, audio_dir=None, audio_files = None):
        # Find path of all audio files 
        if audio_files is None:
            if audio_dir is None:
                raise ValueError("Either of audio_dir and audio_files has to be provided.")
            audio_files = glob(os.path.join(audio_dir, '*.wav')) + glob(os.path.join(audio_dir, '*.mp3'))
        voice_samples = []
        max_length = 0
        
        # Load all audio_files
        for filename in audio_files:
            
            waveform = load_audio(filename, self.input_sample_rate)
            max_length = max(max_length, waveform.shape[-1])
            voice_samples.append(waveform)

        # Pad All to max length
        n_samples = len(voice_samples)
        for i in range(n_samples):
            voice_samples[i] = pad_or_truncate(voice_samples[i], max_length)

        return voice_samples

    @torch.inference_mode()
    def format_conditioning(self, clip, cond_length=132300, 
                            # device='cuda', 
                            # sampling_rate=22050
                            ):
        """
        Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
        """
        gap = clip.shape[-1] - cond_length
        if gap < 0:
            clip = F.pad(clip, pad=(0, abs(gap)))
        elif gap > 0:
            rand_start = random.randint(0, gap)
            clip = clip[:, rand_start:rand_start + cond_length]
        mel_clip = TorchMelSpectrogram(sampling_rate=self.input_sample_rate)(clip.unsqueeze(0)).squeeze(0)
        mel_clip = mel_clip.unsqueeze(0)
        return mel_clip.to(ACC_DEVICE)

        # return migrate_to_device(mel_clip, device)
        
    @torch.inference_mode()
    def get_conditioning_latents(self, 
                                 voice_samples, 
                                 return_mels=False, 
                                 verbose=False, 
                                 slices=1, 
                                 max_chunk_size=None, 
                                #  force_cpu=False, 
                                 original_ar=False, 
                                 original_diffusion=False):
        """
        Transforms one or more voice_samples into a tuple (autoregressive_conditioning_latent, diffusion_conditioning_latent).
        These are expressive learned latents that encode aspects of the provided clips like voice, intonation, and acoustic
        properties.
        :param voice_samples: List of 2 or more ~10 second reference clips, which should be torch tensors containing 22.05kHz waveform data.
        """
        with torch.no_grad():
            # computing conditional latents requires being done on the CPU if using DML because M$ still hasn't implemented some core functions
            if not isinstance(voice_samples, list):
                voice_samples = [voice_samples]
            
            resampler_22K = torchaudio.transforms.Resample(
                self.input_sample_rate,
                22050,
                lowpass_filter_width=16,
                rolloff=0.85,
                resampling_method="kaiser_window",
                beta=8.555504641634386,
            ).to(ACC_DEVICE)

            resampler_24K = torchaudio.transforms.Resample(
                self.input_sample_rate,
                24000,
                lowpass_filter_width=16,
                rolloff=0.85,
                resampling_method="kaiser_window",
                beta=8.555504641634386,
            ).to(ACC_DEVICE)

            # voice_samples = [migrate_to_device(v, device)  for v in voice_samples]

            auto_conds = []
            diffusion_conds = []

            if original_ar:
                samples = [resampler_22K(sample.to(ACC_DEVICE)).detach().to(CPU_DEVICE) for sample in self.voice_samples]
                for sample in tqdm(samples, desc="Computing AR conditioning latents..."):
                    auto_conds.append(self.format_conditioning(sample, 
                                                        #   device=device, 
                                                        #   sampling_rate=self.input_sample_rate, 
                                                          cond_length=132300))
            else:
                samples = [resampler_22K(sample.to(ACC_DEVICE)).detach().to(CPU_DEVICE) for sample in voice_samples]
                concat = torch.cat(samples, dim=-1)
                chunk_size = concat.shape[-1]

                if slices == 0:
                    slices = 1
                elif max_chunk_size is not None and chunk_size > max_chunk_size:
                    slices = 1
                    while int(chunk_size / slices) > max_chunk_size:
                        slices = slices + 1

                chunks = torch.chunk(concat, slices, dim=1)
                chunk_size = chunks[0].shape[-1]

                for chunk in tqdm(chunks, desc="Computing AR conditioning latents..."):
                    auto_conds.append(self.format_conditioning(chunk, 
                                                        #   device=device, 
                                                        #   sampling_rate=self.input_sample_rate, 
                                                          cond_length=chunk_size))
                

            if original_diffusion:
                samples = [resampler_24K(sample.to(ACC_DEVICE)).detach().to(CPU_DEVICE) for sample in voice_samples]
                for sample in tqdm(samples, desc="Computing diffusion conditioning latents..."):
                    sample = pad_or_truncate(sample, 102400)
                    cond_mel = wav_to_univnet_mel(sample, do_normalization=False, device=self.device)
                    diffusion_conds.append(cond_mel)
            else:
                samples = [resampler_24K(sample.to(ACC_DEVICE)).detach().to(CPU_DEVICE) for sample in voice_samples]
                for chunk in tqdm(chunks, desc="Computing diffusion conditioning latents..."):
                    # check_for_kill_signal()
                    chunk = pad_or_truncate(chunk, chunk_size)
                    cond_mel = wav_to_univnet_mel(chunk, do_normalization=False, device=self.device)
                    diffusion_conds.append(cond_mel)

            auto_conds = torch.stack(auto_conds, dim=1)
            print('Entering Autoregressive Processing')
            self.autoregressive = self.autoregressive.to(ACC_DEVICE)
            do_gc()
            auto_latent = self.autoregressive.get_conditioning(auto_conds)
            print('Finished Autoregressive Processing')
            self.autoregressive = self.autoregressive.to(CPU_DEVICE)
            do_gc()
            # self.autoregressive = migrate_to_device( self.autoregressive, self.device if self.preloaded_tensors else 'cpu' )

            diffusion_conds = torch.stack(diffusion_conds, dim=1)
            print('Entering Diffusion Processing')            
            self.diffusion = self.diffusion.to(ACC_DEVICE)
            do_gc()
            # self.diffusion = migrate_to_device( self.diffusion, device )
            diffusion_latent = self.diffusion.get_conditioning(diffusion_conds)            
            self.diffusion = self.diffusion.to(CPU_DEVICE)
            do_gc()
            print('Finished Diffusion Processing')
            # self.diffusion = migrate_to_device( self.diffusion, self.device if self.preloaded_tensors else 'cpu' )

        if return_mels:
            return auto_latent, diffusion_latent, auto_conds, diffusion_conds
        else:
            return auto_latent, diffusion_latent

    def save_conditioning_latents(self, voice, conditioning_latents):
        if len(conditioning_latents) == 4:
            conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
        
        outfile = f'latents/cond_latents_{voice}.pth'
        torch.save(conditioning_latents, outfile)
        print(f'Saved voice latents: {outfile}')


