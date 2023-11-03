import os
import torch
import torchaudio
from glob import glob
from models.tortoise import TortoiseModel

from utils.audio import load_audio, pad_or_truncate

class TortoiseLatent(TortoiseModel):
    def __init__(self):
        super().__init__()
        self.input_sample_rate = 22050

    def load_voice_files(self, audio_dir=None, audio_files = None):
        # Find path of all audio files 
        if audio_files is None:
            if audio_dir is None:
                raise ValueError("Either of audio_dir and audio_files has to be provided.")
            audio_files = glob(os.path.join(audio_dir, '*.wav')) + glob(os.path.join(audio_dir, '*.mp3'))

        self.voice_samples = []
        self.max_length = 0
        # Load all audio_files
        for filename in audio_files:
            
            waveform = load_audio(filename, self.input_sample_rate)
            max_length = max(max_length, waveform.shape[-1])
            self.voice_samples.append(waveform)

        # Pad All to max length
        n_samples = len(self.voice_samples)
        for i in range(n_samples):
            self.voice_samples[i] = pad_or_truncate(self.voice_samples[i], self.sample_rate)

    @torch.inference_mode()
    def get_conditioning_latents(self, voice_samples, return_mels=False, verbose=False, slices=1, max_chunk_size=None, force_cpu=False, original_ar=False, original_diffusion=False):
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
            ).to(device)

            resampler_24K = torchaudio.transforms.Resample(
                self.input_sample_rate,
                24000,
                lowpass_filter_width=16,
                rolloff=0.85,
                resampling_method="kaiser_window",
                beta=8.555504641634386,
            ).to(device)

            voice_samples = [migrate_to_device(v, device)  for v in voice_samples]

            auto_conds = []
            diffusion_conds = []

            if original_ar:
                samples = [resampler_22K(sample) for sample in voice_samples]
                for sample in tqdm(samples, desc="Computing AR conditioning latents..."):
                    auto_conds.append(format_conditioning(sample, 
                                                          device=device, 
                                                          sampling_rate=self.input_sample_rate, 
                                                          cond_length=132300))
            else:
                samples = [resampler_22K(sample) for sample in voice_samples]
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
                    auto_conds.append(format_conditioning(chunk, 
                                                          device=device, 
                                                          sampling_rate=self.input_sample_rate, 
                                                          cond_length=chunk_size))
                

            if original_diffusion:
                samples = [resampler_24K(sample) for sample in voice_samples]
                for sample in tqdm(samples, desc="Computing diffusion conditioning latents..."):
                    sample = pad_or_truncate(sample, 102400)
                    cond_mel = wav_to_univnet_mel(migrate_to_device(sample, device), do_normalization=False, device=self.device)
                    diffusion_conds.append(cond_mel)
            else:
                samples = [resampler_24K(sample) for sample in voice_samples]
                for chunk in tqdm(chunks, desc="Computing diffusion conditioning latents..."):
                    check_for_kill_signal()
                    chunk = pad_or_truncate(chunk, chunk_size)
                    cond_mel = wav_to_univnet_mel(migrate_to_device( chunk, device ), do_normalization=False, device=device)
                    diffusion_conds.append(cond_mel)

            auto_conds = torch.stack(auto_conds, dim=1)
            self.autoregressive = migrate_to_device( self.autoregressive, device )
            auto_latent = self.autoregressive.get_conditioning(auto_conds)
            self.autoregressive = migrate_to_device( self.autoregressive, self.device if self.preloaded_tensors else 'cpu' )

            diffusion_conds = torch.stack(diffusion_conds, dim=1)
            self.diffusion = migrate_to_device( self.diffusion, device )
            diffusion_latent = self.diffusion.get_conditioning(diffusion_conds)
            self.diffusion = migrate_to_device( self.diffusion, self.device if self.preloaded_tensors else 'cpu' )

        if return_mels:
            return auto_latent, diffusion_latent, auto_conds, diffusion_conds
        else:
            return auto_latent, diffusion_latent



