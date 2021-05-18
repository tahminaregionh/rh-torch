import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
import numpy as np


def double_conv(in_c, out_c, dropout=.2):
    conv = nn.Sequential(
        # conv 1
        nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        # PyTorch initialises weights based on the non-linearity used after the Conv Layer: Kaiming He for ReLU
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
        # nn.Dropout3d(dropout, inplace=True),
        nn.Dropout(dropout),
        # conv 2
        nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout)
    )
    return conv


def downsample_conv(in_c, out_c,kernel_size=3,stride=2):
    downsample = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='replicate'),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True)
    )
    return downsample
    

def up_convt(in_c, out_c,kernel_size=3, stride=2):
    convt = nn.Sequential(
        nn.ConvTranspose3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True)
    )
    return convt
    

class UNet3DMaxPool(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Encoder layers
        self.max_pool3d = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.dconv1 = double_conv(in_channels, 64)
        self.dconv2 = double_conv(64, 128)
        self.dconv3 = double_conv(128, 256)
        self.dconv4 = double_conv(256, 512)
        self.dconv5 = double_conv(512, 1024)
        
        # Decoder layers
        self.up_trans1 = up_convt(1024, 512)
        self.up_conv1 = double_conv(1024, 512)
        self.up_trans2 = up_convt(512, 256)
        self.up_conv2 = double_conv(512, 256)
        self.up_trans3 = up_convt(256, 128)
        self.up_conv3 = double_conv(256, 128)
        self.up_trans4 = up_convt(128, 64)
        self.up_conv4 = double_conv(128, 64)
        
        # Output image
        self.out = nn.Conv3d(in_channels=64,
                             out_channels=1,
                             kernel_size=3, 
                             padding=1, 
                             padding_mode='replicate')

    def forward(self, image):
        """ image.size: (Batch size, Color channels, Depth, Height, Width) """
        
        # ENCODER
        x1 = self.dconv1(image)
        x2 = self.max_pool3d(x1)

        x3 = self.dconv2(x2)
        x4 = self.max_pool3d(x3)

        x5 = self.dconv3(x4)
        x6 = self.max_pool3d(x5)

        x7 = self.dconv4(x6)
        x8 = self.max_pool3d(x7)
        
        x9 = self.dconv5(x8)

        # DECODER
        x = self.up_trans1(x9)
        x = self.up_conv1(torch.cat([x, x7], dim=1))
        
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x5], dim=1))

        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x3], dim=1))

        x = self.up_trans4(x)
        x = self.up_conv4(torch.cat([x, x1], dim=1))

        return self.out(x)
    
    
class UNet3DFullConv(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Encoder layers
        self.dconv1 = double_conv(in_channels, 64)
        self.downs1 = downsample_conv(64, 128)
        
        self.dconv2 = double_conv(128, 128)
        self.downs2 = downsample_conv(128, 256)
        
        self.dconv3 = double_conv(256, 256)
        self.downs3 = downsample_conv(256, 512)
        
        self.dconv4 = double_conv(512, 512)
        self.downs4 = downsample_conv(512, 1024)
        
        # bottom layer
        self.dconv5 = double_conv(1024, 1024)
        
        # Decoder layers
        self.up_trans1 = up_convt(1024, 512)
        self.up_conv1 = double_conv(1024, 512)
        self.up_trans2 = up_convt(512, 256)
        self.up_conv2 = double_conv(512, 256)
        self.up_trans3 = up_convt(256, 128)
        self.up_conv3 = double_conv(256, 128)
        self.up_trans4 = up_convt(128, 64)
        self.up_conv4 = double_conv(128, 64)
        
        # Output image
        self.out = nn.Conv3d(in_channels=64,
                             out_channels=1,
                             kernel_size=3, 
                             padding=1, 
                             padding_mode='replicate')

    def forward(self, image):
        """ image.size: (Batch size, Color channels, Depth, Height, Width) """
        
        # ENCODER
        x1 = self.dconv1(image)
        x2 = self.downs1(x1)

        x3 = self.dconv2(x2)
        x4 = self.downs2(x3)

        x5 = self.dconv3(x4)
        x6 = self.downs3(x5)

        x7 = self.dconv4(x6)
        x8 = self.downs4(x7)
        
        x9 = self.dconv5(x8)

        # DECODER
        x = self.up_trans1(x9)
        x = self.up_conv1(torch.cat([x, x7], dim=1))
        
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, x5], dim=1))

        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, x3], dim=1))

        x = self.up_trans4(x)
        x = self.up_conv4(torch.cat([x, x1], dim=1))

        return self.out(x)


class AEFlatpseudo2D(nn.Module):
    
    def __init__(self, in_channels=1):
        super().__init__()
        
        self.first = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, padding_mode='replicate'),
                                   nn.ReLU(inplace=True))
        self.conv_3d = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode='replicate'),
                                     nn.ReLU(inplace=True))
        self.conv_2d = nn.Sequential(nn.Conv3d(32, 32, kernel_size=(1,3,3), padding=(0,1,1), padding_mode='replicate'),
                                     nn.ReLU(inplace=True))
        
        self.dconv_3d = nn.Sequential(nn.ConvTranspose3d(32, 32, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.dconv_2d = nn.ConvTranspose3d(32, 32, kernel_size=(1,3,3), padding=(0,1,1))
        self.last = nn.ConvTranspose3d(32, 1, kernel_size=3, padding=1)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, image):
        """ image.size: (Batch size, Color channels, Depth, Height, Width) """
        
        # ENCODER
        x1 = self.first(image)
        x2 = self.conv_3d(x1)
        # skip here
        
        x3 = self.conv_2d(x2)
        x4 = self.conv_2d(x3)
        # skip here
        x5 = self.conv_2d(x4)
        
        x6 = self.dconv_2d(x5)
        x6 = self.ReLU(x6+x4)
        
        x7 = self.dconv_2d(x6)
        x7 = self.ReLU(x7)
        
        x8 = self.dconv_3d(x7)
        x8 = self.ReLU(x7+x2)
        
        x9 = self.dconv_3d(x8)
        
        x10 = self.last(x9)
        
        return self.ReLU(x10+image)


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm3d(input_dim),
            nn.ReLU(),
            nn.Conv3d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(),
            nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose3d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Res3DUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512], do_sigmoid=False):
        super(Res3DUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        if do_sigmoid:
            self.output_layer = nn.Sequential(
                nn.Conv3d(filters[0], 1, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Conv3d(filters[0], 1, 1, 1),
            )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
