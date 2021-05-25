import torch
import torch.nn as nn


class Block(nn.Module):
    """ Double conv layer used as feature extractor in encoder/decoder"""

    def __init__(self, in_c, out_c, dropout=.2, f_act='ReLU'):
        super().__init__()
        activation_func = getattr(nn, f_act)
        self.double_conv = nn.Sequential(
            # conv 1
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1,
                      padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(out_c),
            activation_func(inplace=True),
            nn.Dropout(dropout),
            # conv 2
            nn.Conv3d(out_c, out_c, kernel_size=3, stride=1,
                      padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(out_c),
            activation_func(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownsamplingBlock(nn.Module):
    """ Downsampling block using a Convolutional layer with stride=2 """

    def __init__(self, in_c, out_c, kernel_size=3, stride=2, f_act='ReLU'):
        super().__init__()
        activation_func = getattr(nn, f_act)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size,
                      stride=stride, padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(out_c),
            activation_func(inplace=True)
        )

    def forward(self, x):
        return self.downsample(x)


class UpsamplingBlock(nn.Module):
    """ Upsampling block obtained from ConvTranspose3D layer """

    def __init__(self, in_c, out_c, kernel_size=3, stride=2, f_act='ReLU'):
        super().__init__()
        activation_func = getattr(nn, f_act)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=kernel_size,
                               stride=stride, padding=1, output_padding=1),
            nn.BatchNorm3d(out_c),
            activation_func(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, chs, pooling, f_act):
        super().__init__()
        self.chs = (in_channels, *chs)
        self.depth = len(self.chs) - 1
        self.enc_blocks = nn.ModuleList(
            [Block(self.chs[i], self.chs[i+1], f_act=f_act) for i in range(self.depth)])
        if pooling == 'max_pool':
            self.pool = nn.ModuleList(
                [nn.MaxPool3d(kernel_size=3, stride=2, padding=1)] * (self.depth))
        elif pooling == 'full_conv':
            self.pool = nn.ModuleList(
                [DownsamplingBlock(c, c, f_act=f_act) for c in chs])

    def forward(self, x):
        ftrs = []
        for block, pool_block in zip(self.enc_blocks, self.pool):
            x = block(x)
            ftrs.append(x)
            x = pool_block(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs, f_act):
        super().__init__()
        self.depth = len(chs) - 1
        self.upconvs = nn.ModuleList(
            [UpsamplingBlock(chs[i], chs[i+1], f_act=f_act) for i in range(self.depth)])
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1], f_act=f_act) for i in range(self.depth)])

    def forward(self, x, encoder_features):
        for i in range(self.depth):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels, g_filters=[64, 128, 256, 512, 1024], g_pooling_type='full_conv', g_activation='ReLU', **kwargs):
        super().__init__()
        # decoding channels dims are the same as encoding channels but inverted
        dec_chs = g_filters[::-1]
        self.encoder = Encoder(in_channels, g_filters, g_pooling_type, g_activation)
        self.decoder = Decoder(dec_chs, g_activation)
        self.head = nn.Conv3d(
            in_channels=dec_chs[-1], out_channels=1, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        enc_ftrs = enc_ftrs[::-1]
        out = self.decoder(enc_ftrs[0], enc_ftrs[1:])
        return self.head(out)

"""
UNet-like network that never uses striding, so no downsampling is performed.
"""
class AEFlatpseudo2D(nn.Module):

    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        self.first = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, padding_mode='replicate'),
                                   nn.ReLU(inplace=True))
        self.conv_3d = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode='replicate'),
                                     nn.ReLU(inplace=True))
        self.conv_2d = nn.Sequential(nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='replicate'),
                                     nn.ReLU(inplace=True))

        self.dconv_3d = nn.Sequential(nn.ConvTranspose3d(32, 32, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.dconv_2d = nn.ConvTranspose3d(
            32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))
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


class Res3DUnet(nn.Module):
    def __init__(self, in_channels, g_filters=[64, 128, 256, 512], do_sigmoid=False, **kwargs):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, g_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(g_filters[0]),
            nn.ReLU(),
            nn.Conv3d(g_filters[0], g_filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(in_channels, g_filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = self.ResidualConv(g_filters[0], g_filters[1], 2, 1)
        self.residual_conv_2 = self.ResidualConv(g_filters[1], g_filters[2], 2, 1)

        self.bridge = self.ResidualConv(g_filters[2], g_filters[3], 2, 1)

        self.upsample_1 = self.Upsample(g_filters[3], g_filters[3], 2, 2)
        self.up_residual_conv1 = self.ResidualConv(
            g_filters[3] + g_filters[2], g_filters[2], 1, 1)

        self.upsample_2 =self.Upsample(g_filters[2], g_filters[2], 2, 2)
        self.up_residual_conv2 = self.ResidualConv(
            g_filters[2] + g_filters[1], g_filters[1], 1, 1)

        self.upsample_3 = self.Upsample(g_filters[1], g_filters[1], 2, 2)
        self.up_residual_conv3 = self.ResidualConv(
            g_filters[1] + g_filters[0], g_filters[0], 1, 1)

        if do_sigmoid:
            self.output_layer = nn.Sequential(
                nn.Conv3d(g_filters[0], 1, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Conv3d(g_filters[0], 1, 1, 1),
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
    
    class ResidualConv(nn.Module):
        def __init__(self, input_dim, output_dim, stride, padding):
            super().__init__()
    
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
                nn.Conv3d(input_dim, output_dim, kernel_size=3,
                          stride=stride, padding=1),
                nn.BatchNorm3d(output_dim),
            )
    
        def forward(self, x):
    
            return self.conv_block(x) + self.conv_skip(x)
    
    
    class Upsample(nn.Module):
        def __init__(self, input_dim, output_dim, kernel, stride):
            super().__init__()
    
            self.upsample = nn.ConvTranspose3d(
                input_dim, output_dim, kernel_size=kernel, stride=stride
            )
    
        def forward(self, x):
            return self.upsample(x)
