import json

import torch
from torch import nn
# from torch.nn.utils import weight_norm, 

from models.pieces.vocoder_pieces import init_weights, LVCBlock, AttrDict, AMPBlock1, AMPBlock2, Activation1d, Snake, SnakeBeta

class UnivNetGenerator(nn.Module):
    """
    UnivNet Generator
    
    Originally from https://github.com/mindslab-ai/univnet/blob/master/model/generator.py.
    """

    def __init__(self, noise_dim=64, channel_size=32, dilations=[1,3,9,27], strides=[8,8,4], lReLU_slope=.2, kpnet_conv_size=3,
                 # Below are MEL configurations options that this generator requires.
                 hop_length=256, n_mel_channels=100):
        super(UnivNetGenerator, self).__init__()
        self.mel_channel = n_mel_channels
        self.noise_dim = noise_dim
        self.hop_length = hop_length
        channel_size = channel_size
        kpnet_conv_size = kpnet_conv_size

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    n_mel_channels,
                    stride=stride,
                    dilations=dilations,
                    lReLU_slope=lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )

        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def forward(self, c, z):
        '''
        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)
            z (Tensor): the noise sequence (batch, noise_dim, in_length)

        '''
        z = self.conv_pre(z)  # (B, c_g, L)

        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c)  # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)  # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(UnivNetGenerator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((c.shape[0], self.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)

        if z is None:
            z = torch.randn(c.shape[0], self.noise_dim, mel.size(2)).to(mel.device)

        audio = self.forward(mel, z)
        audio = audio[:, :, :-(self.hop_length * 10)]
        audio = audio.clamp(min=-1, max=1)
        return audio
    


class BigVGAN(nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, config=None, data=None):
        super(BigVGAN, self).__init__()

        """
        with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
            data = f.read()
        """
        if config and data is None:
            with open(config, 'r') as f:
                data = f.read()
            jsonConfig = json.loads(data)
        elif data is not None:
            if isinstance(data, str):
                jsonConfig = json.loads(data)
            else:
                jsonConfig = data
        else:
            raise Exception("no config specified")


        global h
        h = AttrDict(jsonConfig)

        self.mel_channel = h.num_mels
        self.noise_dim = h.n_fft
        self.hop_length = h.hop_size
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = nn.utils.weight_norm(nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                nn.utils.weight_norm(nn.ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # post conv
        if h.activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self,x, c):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                nn.utils.remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)

    def inference(self, c, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((c.shape[0], h.num_mels, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)

        if z is None:
            z = torch.randn(c.shape[0], self.noise_dim, mel.size(2)).to(mel.device)

        audio = self.forward(mel, z)
        audio = audio[:, :, :-(self.hop_length * 10)]
        audio = audio.clamp(min=-1, max=1)
        return audio

    def eval(self, inference=False):
        super(BigVGAN, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()


