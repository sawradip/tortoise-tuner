
import torch
import torchaudio
from torch import nn

DEFAULT_MEL_NORM_FILE = "utils/data/mel_norms.pth"
# os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/mel_norms.pth')


class TorchMelSpectrogram(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
                 sampling_rate=22050, normalize=False, mel_norm_file=DEFAULT_MEL_NORM_FILE):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel
