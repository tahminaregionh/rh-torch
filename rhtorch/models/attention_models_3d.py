import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fftn, ifftn
# import torch_dct as dct


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 dilation=1):
        super().__init__()
        block = []
        block.append(
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=True,
                      dilation=dilation))
        block.append(nn.PReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SAB_astrous(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        block = []
        block.append(
            ConvBlock(in_channels=in_channels, out_channels=in_channels))
        out_stage1 = in_channels // reduction
        block.append(
            ConvBlock(in_channels=in_channels,
                      out_channels=out_stage1,
                      kernel_size=1,
                      padding=0))
        block.append(
            ConvBlock(in_channels=out_stage1,
                      out_channels=out_stage1,
                      kernel_size=3,
                      padding=2,
                      dilation=2))
        out_stage2 = out_stage1 // reduction
        block.append(
            ConvBlock(in_channels=out_stage1,
                      out_channels=out_stage2,
                      kernel_size=1,
                      padding=0))
        block.append(
            ConvBlock(in_channels=out_stage2,
                      out_channels=out_stage2,
                      kernel_size=3,
                      padding=4,
                      dilation=4))
        block.append(
            nn.Conv3d(out_stage2, 1, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention part
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        block = []
        block.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        block.append(
            ConvBlock(in_channels=channels,
                      out_channels=channels // reduction,
                      kernel_size=1,
                      padding=0))
        block.append(
            nn.Conv3d(channels // reduction,
                      channels,
                      kernel_size=1,
                      padding=0,
                      bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SCAB(nn.Module):
    """
    Dual attention block
    """

    def __init__(self, org_channels, out_channels):
        super().__init__()
        pre_x = []
        pre_x.append(
            ConvBlock(in_channels=org_channels, out_channels=out_channels))
        pre_x.append(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))

        self.pre_x = nn.Sequential(*pre_x)
        self.CAB = ChannelAttention(channels=out_channels)
        self.SAB = SAB_astrous(in_channels=out_channels)
        self.last = nn.Conv3d(in_channels=2 * out_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1,
                              bias=True)

    def forward(self, x):
        pre_x = self.pre_x(x)
        channel = self.CAB(pre_x)
        spatial = self.SAB(pre_x)
        out_s = pre_x * spatial.expand_as(pre_x)
        out_c = pre_x * channel.expand_as(pre_x)
        out_combine = torch.cat([out_s, out_c], dim=1)
        out = self.last(out_combine)
        out = x + out
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_size, out_size, shortcut=None):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.PReLU())
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return out


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, num):
        super().__init__()
        block = []

        block.append(
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.PReLU())

        for _ in range(max(num - 1, 1)):
            block.append(ResidualBlock(out_size, out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, num):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_size,
                                     out_size,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, num)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):

    def __init__(self,
                 in_channels=5,
                 out_channels=4,
                 depth=4,
                 feature_dims=64):
        super().__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, (2**i) * feature_dims, depth - i))
            if i != depth - 1:
                self.down_path.append(
                    SCAB((2**i) * feature_dims, (2**i) * feature_dims))
            prev_channels = (2**i) * feature_dims

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, (2**i) * feature_dims, depth - i))
            prev_channels = (2**i) * feature_dims

        self.last = nn.Conv3d(prev_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if (i != len(self.down_path) - 1) and (i % 2 == 1):
                blocks.append(x)
                x = F.avg_pool3d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class SigmaNet(nn.Module):

    def __init__(self, in_channels, out_channels, depth=3, num_filter=64):
        super().__init__()
        block = []
        block.append(
            nn.Conv3d(in_channels,
                      num_filter,
                      kernel_size=3,
                      padding=1,
                      bias=True))
        block.append(nn.PReLU())

        for _ in range(depth):
            block.append(
                nn.Conv3d(num_filter,
                          num_filter,
                          kernel_size=3,
                          padding=1,
                          bias=True))
            block.append(nn.PReLU())

        block.append(
            nn.Conv3d(num_filter,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class FAN3D(nn.Module):
    """ 
        Frequency Attention Network in 3D
        
        Implementation of UNET using attention blocks with 3D input 
        and working in the frequency domain via the Fourier transform.
        Works for PET, PEt+CT or PET+MR or PET+CT+MR input.
        Activation functions throughout are PReLU and no BN, no dropout.
        This is a Black & White implementation of https://github.com/momo1689/FAN 
                
    """

    def __init__(self,
                 in_channels,
                 depth_S=5,
                 depth_U=4,
                 feature_dims=64,
                 **kwargs):
        super().__init__()
        self.sigma_net = SigmaNet(in_channels=1,
                                  out_channels=1,
                                  depth=depth_S,
                                  num_filter=feature_dims)
        self.UNet = UNet(in_channels=in_channels + 3,
                         out_channels=2,
                         depth=depth_U,
                         feature_dims=feature_dims)

    def forward(self, img):
        # get noise map from SigmaNet. Only pass PET image
        if self.in_channels == 1:
            noise_map = self.sigma_net(img)
        else:
            noise_map = self.sigma_net(img[:, :1, ...])

        # Fourier decomposition on gray scale image. Split REAL and IMAG part
        fourier_decomp = fftn(img)
        f_real = fourier_decomp.real
        f_imag = fourier_decomp.imag

        # concat (Fourier decomposition + image + noise_map) as input to UNET
        # (bs, 4, dim1, dim2)
        net_input = torch.cat((f_real, f_imag, img, noise_map), dim=1)

        # run through the UNet
        net_out = self.UNet(net_input)
        # extract REAL and IMAG part
        out_fftr = net_out[:, 0].unsqueeze(dim=1)
        out_ffti = net_out[:, 1].unsqueeze(dim=1)

        # compute the inverse FFT to generate output image
        out_img = torch.abs(ifftn(out_fftr + 1j * out_ffti))

        # return self.out(out_final)
        if torch.any(torch.isnan(out_img)):
            raise ValueError(
                "Network output contains nan values. Something's wrong.")
        return out_img


class AttentionUNet(nn.Module):
    """ 
        Attention UNet in 3D
        
        Implementation of UNET using attention blocks with 3D input 
        can be PET only, or PET+CT or PET+MR
        Activation functions throughout are PReLU and no BN, no dropout.
        This is a re-work of https://github.com/momo1689/FAN 
                
    """

    def __init__(self,
                 in_channels,
                 depth_S=5,
                 depth_U=4,
                 feature_dims=64,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.sigma_net = SigmaNet(in_channels=1,
                                  out_channels=1,
                                  depth=depth_S,
                                  num_filter=feature_dims)
        self.UNet = UNet(in_channels=in_channels + 1,
                         out_channels=1,
                         depth=depth_U,
                         feature_dims=feature_dims)

    def forward(self, img):
        # get noise map from SigmaNet. Only pass PET image
        if self.in_channels == 1:
            noise_map = self.sigma_net(img)
        else:
            noise_map = self.sigma_net(img[:, :1, ...])

        # concat (Fourier decomposition + image + noise_map) as input to UNET
        # (bs, 4, dim1, dim2)
        net_input = torch.cat((img, noise_map), dim=1)

        # run through the UNet
        out_img = self.UNet(net_input)

        if torch.any(torch.isnan(out_img)):
            raise ValueError(
                "Network output contains nan values. Something's wrong.")
        return out_img


class FAN3D(nn.Module):
    """ 
        Frequency attention Network in 3D with Fourier Transform.
        Same as FAN2D (but for 3D!) in attention_models_2d.py
        
        Implementation of UNET using attention blocks with 3D input 
        can be PET only, or PET+CT or PET+MR
        Activation functions throughout are PReLU and no BN, no dropout.
        This is a re-work of https://github.com/momo1689/FAN
        where I changed wavelet transform to Fourier transform.
        
        WARNING: I have had mixed results with this network.
                 Mostly I get exploding gradients because of no BN.
                 So need careful choice of learning rate scheduler.
                
    """

    def __init__(self,
                 in_channels,
                 depth_S=5,
                 depth_U=4,
                 feature_dims=64,
                 **kwargs):
        super().__init__()
        self.sigma_net = SigmaNet(in_channels=1,
                                  out_channels=1,
                                  depth=depth_S,
                                  num_filter=feature_dims)
        self.UNet = UNet(in_channels=in_channels + 3,
                         out_channels=2,
                         depth=depth_U,
                         feature_dims=feature_dims)

    def forward(self, img):
        # get noise map from SigmaNet. Only pass PET image
        if self.in_channels == 1:
            noise_map = self.sigma_net(img)
        else:
            noise_map = self.sigma_net(img[:, :1, ...])

        # Fourier decomposition on gray scale image. Split REAL and IMAG part
        fourier_decomp = fftn(img)
        f_real = fourier_decomp.real
        f_imag = fourier_decomp.imag

        # concat (Fourier decomposition + image + noise_map) as input to UNET
        # (bs, 4, dim1, dim2)
        net_input = torch.cat((f_real, f_imag, img, noise_map), dim=1)

        # run through the UNet
        net_out = self.UNet(net_input)
        # extract REAL and IMAG part
        out_fftr = net_out[:, 0].unsqueeze(dim=1)
        out_ffti = net_out[:, 1].unsqueeze(dim=1)

        # compute the inverse FFT to generate output image
        out_img = torch.abs(ifftn(out_fftr + 1j * out_ffti))

        if torch.any(torch.isnan(out_img)):
            raise ValueError(
                "Network output contains nan values. Something's wrong.")
        return out_img