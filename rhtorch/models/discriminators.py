# -*- coding: utf-8 -*-
import torch
#from caai.pytorchlightning.models import autoencoders
#from . import autoencoders
import torch.nn as nn


class ConvNetDiscriminator(nn.Module):
    def __init__(self, in_channels=2, dis_f=64, **kwargs):
        super().__init__()
        """ Defines a PatchGAN discriminator
        
        Parameters
        ----------
        in_channels : int, optional
            input_data should contain 2 channels from concatenating 
            the real + fake image. Default is 2.
        dis_f : int, optional
            Number of filters in the first conv layer. Default is 64.

        """
        # Determine the number of channels used for the discriminator.
        if in_channels < 2:
            # In channels must be at least 2 (input (conditioned image) and target (fake or real))
            disc_in_channels = 2
            self.d_input_index = [0]
        elif 'd_input_index' in kwargs:
            disc_in_channels = len(kwargs['d_input_index'])+1
            self.d_input_index = kwargs['d_input_index']
        else:
            # Default is to pick the first color channel only
            disc_in_channels = 2
            self.d_input_index = [0]
        
        # weight initialization - may experiment with kaiming
        # init = normal_(stddev=0.02)	
        self.dis_f = dis_f

        self.c1 = self.conv_layer(disc_in_channels, self.dis_f, bn=False)
        self.c2 = self.conv_layer(self.dis_f, self.dis_f*2)
        self.c3 = self.conv_layer(self.dis_f*2, self.dis_f*4)
        # in pix2pix the last down sampling is treated separately with ZeroPadding3D
        self.c4 = self.conv_layer(self.dis_f*4, self.dis_f*8)

        # patch output NxN. Use BCEwithlogits loss on this output
        # !!! changes kernel_size to 3 instead of 4 because dim1=1 at this stage
        # with padding it gives 3 pixels, so cannot use kernel with size 4
        self.out = nn.Conv3d(in_channels=self.dis_f*8, out_channels=1, kernel_size=3, padding=1, stride=1)
        
    @staticmethod
    def conv_layer(in_c, out_c, f_size=4, bn=True):
        # out_c = number of filters to use
        layers = []
        layers.append(nn.Conv3d(in_c, out_c, kernel_size=f_size, stride=2, padding=1))
        # PyTorch initialises weights based on the non-linearity used after the Conv Layer: Kaiming He for ReLU
        if bn:
            layers.append(nn.BatchNorm3d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, inp, tar):
        """ xs = list(input_image, real_or_fake_target_image)
            shape = (bs, cc, dim1, dim2, dim3)"""
        bs, cc, d1, d2, d3 = list(tar.size())

        # Select the first selected color channel (0 if nothing else is selected)
        selected_inp = inp[:, self.d_input_index[0], ...].view(bs, 1, d1, d2, d3)
        # Add other color channels if applicable
        if len(self.d_input_index) > 1:
            for i in range(1, len(self.d_input_index)):
                _selected_inp = inp[:, self.d_input_index[i], ...].view(bs, 1, d1, d2, d3)
                selected_inp = torch.cat([selected_inp, _selected_inp], dim=1)        

        x = torch.cat([selected_inp, tar], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return self.out(x)

        

