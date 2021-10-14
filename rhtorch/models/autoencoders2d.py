import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


########################################################################################################
##################################   WAVELET TRANSFORM FOR PYTORCH   ###################################
########################################################################################################
 
def wave_pad(x, padding, mode='periodic', value=0):
    if mode == 'periodic':
        # only vertical
        if padding[0] == 0 and padding[1] == 0:
            x_pad = np.arange(x.shape[-2])
            x_pad = np.pad(x_pad, (padding[2], padding[3]), mode='wrap')
            return x[:, :, x_pad, :]
        # only horizontal
        elif padding[2] == 0 and padding[3] == 0:
            x_pad = np.arange(x.shape[-1])
            x_pad = np.pad(x_pad, (padding[0], padding[1]), mode='wrap')
            return x[:, :, :, x_pad]
        # both
        else:
            x_pad_col = np.arange(x.shape[-2])
            x_pad_col = np.pad(x_pad_col, (padding[2], padding[3]), mode='wrap')
            x_pad_row = np.arange(x.shape[-1])
            x_pad_row = np.pad(x_pad_row, (padding[0], padding[1]), mode='wrap')
            col = np.outer(x_pad_col, np.ones(x_pad_row.shape[0]))
            row = np.outer(np.ones(x_pad_col.shape[0]), x_pad_row)
            return x[:, :, col, row]
    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, padding, mode, value)
    else:
        raise ValueError("Unknown padding mode".format(mode))


def prep_filt(low_col, high_col, low_row=None, high_row=None):
    low_col = np.array(low_col).ravel()
    high_col = np.array(high_col).ravel()
    if low_row is None:
        low_row = low_col
    else:
        low_row = np.array(low_row).ravel()
    if high_row is None:
        high_row = high_col
    else:
        high_row = np.array(high_row).ravel()
    low_col = torch.from_numpy(low_col).reshape((1, 1, -1, 1)).float()
    high_col = torch.from_numpy(high_col).reshape((1, 1, -1, 1)).float()
    low_row = torch.from_numpy(low_row).reshape((1, 1, 1, -1)).float()
    high_row = torch.from_numpy(high_row).reshape((1, 1, 1, -1)).float()

    return low_col, high_col, low_row, high_row


def upsample(filts):
    new_filts = []
    for f in filts:
        if f.shape[3] == 1:
            new = torch.zeros((f.shape[0], f.shape[1], 2*f.shape[2], f.shape[3]), dtype=torch.float, device=f.device)
            new[:, :, ::2, :] = f.clone()
        else:
            new = torch.zeros((f.shape[0], f.shape[1], f.shape[2], 2*f.shape[3]), dtype=torch.float, device=f.device)
            new[:, :, :, ::2] = f.clone()
        new_filts.append(new)
    return new_filts


def afb1d(x_pad, low, high, dim):
    """
    :param x: Tensor (N, C, H, W)
    :param low: low-pass filter
    :param high: high-pass filter
    :param dilation:
    :return:
        low: Tensor (N, C, H, W)
        high: Tensor (N, C, H, W)
    """
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(np.copy(np.array(low).ravel()[::-1]),
                           dtype=torch.float, device=low.device)
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(np.copy(np.array(high).ravel()[::-1]),
                            dtype=torch.float, device=high.device)
    shape = [1, 1, 1, 1]
    shape[dim] = low.numel()
    # If filter aren't in the right shape, make them so
    if low.shape != tuple(shape):
        low = low.reshape(*shape)
    if high.shape != tuple(shape):
        high = high.reshape(*shape)

    low_band = F.conv2d(x_pad, low)
    high_band = F.conv2d(x_pad, high)

    return low_band, high_band


def afb2d(x, filts, mode):
    """
    :param x: Tensor (N, C, H, W)
    :param filts:
    :param mode:
    :param dilation:
    :return:
        cA, cH, cV, cD: Tensor (N, C, H, W) four sub bands
    """
    low_col = filts[0].float()
    high_col = filts[1].float()
    low_row = filts[2].float()
    high_row = filts[3].float()
    pad_size = low_col.numel() // 2

    channel = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    if channel == 1:
        padding = (pad_size, pad_size, pad_size, pad_size)
    elif channel == 3:
        padding = (pad_size, pad_size, pad_size, pad_size, 0, 0)
    else:
        raise ValueError('channel should be 1 or 3')
    x_pad = wave_pad(x, padding, mode)
    x_pad = wave_pad(x_pad, padding, 'constant')
    low, high = afb1d(x_pad, low_row, high_row, dim=3)
    cA, cH = afb1d(low, low_col, high_col, dim=2)
    cA = cA[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    cH = cH[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    cV, cD = afb1d(high, low_col, high_col, dim=2)
    cV = cV[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    cD = cD[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    return cA, cH, cV, cD


def upconvLOC(band, row_f, col_f, mode='periodic'):
    """
    :param band: Tensor (N, C, H, W)
    :param row_f: row filter
    :param col_f: col filter
    :param mode: default as 'reflect'
    :return:
    """
    height = band.shape[2]
    width = band.shape[3]
    length = row_f.numel()
    pad_size = length // 2
    padding = (pad_size, pad_size, pad_size, pad_size)
    band_pad = wave_pad(band, padding, mode)
    col_recon = F.conv2d(band_pad, col_f)
    recon = F.conv2d(col_recon, row_f)
    recon = recon[:, :, :height, :width]
    return recon


def idwt2LOC(cA, cH, cV, cD, low_row, high_row, low_col, high_col, mode, shift):
    up_a = upconvLOC(cA, low_row, low_col, mode)
    up_h = upconvLOC(cH, low_row, high_col, mode)
    up_v = upconvLOC(cV, high_row, low_col, mode)
    up_d = upconvLOC(cD, high_row, high_col, mode)
    result = up_a + up_h + up_v + up_d
    if shift:
        # col shift
        result1 = torch.zeros(result.shape, device=cA.device)
        temp_row = result[:, :, -1, :].clone()
        result1[:, :, 1:, :] = result[:, :, :-1, :]
        result1[:, :, 0, :] = temp_row

        # row shift
        result2 = torch.zeros(result.shape, device=cA.device)
        temp_col = result1[:, :, :, -1].clone()
        result2[:, :, :, 1:] = result1[:, :, :, :-1]
        result2[:, :, :, 0] = temp_col
        result = result2
    return result


def sfb2d(ll, high_coeffs, filts, level, mode):
    """
    :param ll: cA Tensor (N, C, H, W)
    :param high_coeffs: (cH, cV, cD) Tensor (N, C, H, W)
    :param filts: filters
    :param level: reconstruction level
    :param mode: default 'reflect'
    :return:
        result: Reconstruction result Tensor (N, C, H, W)
    """
    low_col = filts[0].clone().reshape((1, 1, -1, 1)).float()
    high_col = filts[1].clone().reshape((1, 1, -1, 1)).float()
    low_row = filts[2].clone().reshape((1, 1, 1, -1)).float()
    high_row = filts[3].clone().reshape((1, 1, 1, -1)).float()

    step = 2 ** level
    result = ll.clone()
    for i in range(step):
        gap = step * 2
        new_shape = [ll.shape[0], ll.shape[1], ll.shape[2]//step, ll.shape[3]//step]
        cA1 = torch.zeros(new_shape, device=ll.device)
        cA1[:, :, ::2, ::2] = ll[:, :, i::gap, i::gap]
        cH1 = torch.zeros(new_shape, device=ll.device)
        cH1[:, :, ::2, ::2] = high_coeffs[:, 0, i::gap, i::gap].unsqueeze(dim=1)
        cV1 = torch.zeros(new_shape, device=ll.device)
        cV1[:, :, ::2, ::2] = high_coeffs[:, 1, i::gap, i::gap].unsqueeze(dim=1)
        cD1 = torch.zeros(new_shape, device=ll.device)
        cD1[:, :, ::2, ::2] = high_coeffs[:, 2, i::gap, i::gap].unsqueeze(dim=1)
        shift = False
        out1 = idwt2LOC(cA1, cH1, cV1, cD1, low_row, high_row, low_col, high_col, mode, shift)

        cA2 = torch.zeros(new_shape, device=ll.device)
        cA2[:, :, ::2, ::2] = ll[:, :, step+i::gap, step+i::gap]
        cH2 = torch.zeros(new_shape, device=ll.device)
        cH2[:, :, ::2, ::2] = high_coeffs[:, 0, step+i::gap, step+i::gap].unsqueeze(dim=1)
        cV2 = torch.zeros(new_shape, device=ll.device)
        cV2[:, :, ::2, ::2] = high_coeffs[:, 1, step+i::gap, step+i::gap].unsqueeze(dim=1)
        cD2 = torch.zeros(new_shape, device=ll.device)
        cD2[:, :, ::2, ::2] = high_coeffs[:, 2, step+i::gap, step+i::gap].unsqueeze(dim=1)
        shift = True
        out2 = idwt2LOC(cA2, cH2, cV2, cD2, low_row, high_row, low_col, high_col, mode, shift)
        result[:, :, i::step, i::step] = (out1 + out2) * 0.5
    return result


class SWTForward(nn.Module):
    def __init__(self, wave='db1', level=1, mode='periodic'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            low_col, high_col = wave.dec_lo, wave.dec_hi
            low_row, high_row = low_col, high_col
        else:
            if len(wave) == 2:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = low_col, high_col
            elif len(wave) == 4:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = wave[2], wave[3]

        # Prepare the filters
        low_col, high_col, low_row, high_row = \
            low_col[::-1], high_col[::-1], low_row[::-1], high_row[::-1]
        low_col, high_col, low_row, high_row = \
            prep_filt(low_col, high_col, low_row, high_row)
        # add filters to the network for using F.conv2d (input and weight should be the same dtype)
        self.low_col = nn.Parameter(low_col, requires_grad=False)
        self.high_col = nn.Parameter(high_col, requires_grad=False)
        self.low_row = nn.Parameter(low_row, requires_grad=False)
        self.high_row = nn.Parameter(high_row, requires_grad=False)
        self.mode = mode
        self.level = level

    def forward(self, x):
        """
        :param x: Tensor (N, C, H, W)
        :return: ll: Tensor (N, C, H, W)
                 high_coeffs: List with length of 3*level, each element is Tensor (N, C, H, W)
        """
        filts = [self.low_col, self.high_col, self.low_row, self.high_row]
        ll = x
        high_coeffs = []
        for j in range(self.level):
            y = afb2d(ll, filts, self.mode)
            high_coeffs += y[1:]
            ll = y[0]
            filts = upsample(filts)

        return ll, high_coeffs


class SWTInverse(nn.Module):
    def __init__(self, wave='db1', level=1, mode='periodic'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            low_col, high_col = wave.dec_lo, wave.dec_hi
            low_row, high_row = low_col, high_col
        else:
            if len(wave) == 2:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = low_col, high_col
            elif len(wave) == 4:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = wave[2], wave[3]

        # Prepare the filters
        low_col, high_col, low_row, high_row = \
            prep_filt(low_col, high_col, low_row, high_row)
        self.low_col = nn.Parameter(low_col, requires_grad=False)
        self.high_col = nn.Parameter(high_col, requires_grad=False)
        self.low_row = nn.Parameter(low_row, requires_grad=False)
        self.high_row = nn.Parameter(high_row, requires_grad=False)
        self.level = level
        self.mode = mode

    def forward(self, x):
        """
        :param x: Coeff (ll, high_coeffs)
                  each sub band shape (N, C, H, W)
                  ll: Tensor (N, C, H, W)
                  high_coeffs: Tensor (N, level*3, H, W)
        :return:
                Tensor (N, C, H, W)
        """
        filts = (self.low_col, self.high_col, self.low_row, self.high_row)
        ll = x[0]
        lohi = x[1]
        for i in range(self.level-1, -1, -1):
            lohi_level = lohi[:, i*3:(i+1)*3]
            ll = sfb2d(ll, lohi_level, filts, i, self.mode)
        return ll
    
########################################################################################################
########################################   NETWORK STUFF   #############################################
########################################################################################################
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True, dilation=dilation))
        block.append(nn.PReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention part
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        block = []
        out_stage1 = in_channels // reduction
        out_stage2 = out_stage1 // reduction
        block.append(ConvBlock(in_channels=in_channels, out_channels=out_stage1))
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage2))
        block.append(nn.Conv2d(out_stage2, 1, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SAB_astrous(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        block = []
        block.append(ConvBlock(in_channels=in_channels, out_channels=in_channels))
        out_stage1 = in_channels // reduction
        block.append(ConvBlock(in_channels=in_channels, out_channels=out_stage1, kernel_size=1, padding=0))
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage1, kernel_size=3, padding=2, dilation=2))
        out_stage2 = out_stage1 // reduction
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage2, kernel_size=1, padding=0))
        block.append(ConvBlock(in_channels=out_stage2, out_channels=out_stage2, kernel_size=3, padding=4, dilation=4))
        block.append(nn.Conv2d(out_stage2, 1, kernel_size=1, padding=0, bias=True))
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
        block.append(nn.AdaptiveAvgPool2d((1, 1)))
        block.append(ConvBlock(in_channels=channels, out_channels=channels // reduction, kernel_size=1, padding=0))
        block.append(nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SCAB(nn.Module):
    """
    Dual (Spatial + Channel) attention block
    """
    def __init__(self, org_channels, out_channels):
        super().__init__()
        pre_x = []
        pre_x.append(ConvBlock(in_channels=org_channels, out_channels=out_channels))
        pre_x.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.pre_x = nn.Sequential(*pre_x)
        self.CAB = ChannelAttention(channels=out_channels)
        self.SAB = SAB_astrous(in_channels=out_channels)
        self.last = torch.nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        pre_x = self.pre_x(x)
        # pre_map = self.pre_map(map)
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
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.PReLU()
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, num):
        super().__init__()
        # print('conv block in_size =', in_size)
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.PReLU())

        for i in range(max(num-1, 1)):
            block.append(ResidualBlock(out_size, out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, num):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, num)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, filters=[64, 128, 256, 512, 1024]):
        super().__init__()
        depth = len(filters)
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for j, feature_dims in enumerate(filters):
            # print('input UNetConvBlock:', feature_dims, depth-j)
            self.down_path.append(UNetConvBlock(prev_channels, feature_dims, depth-j))
            if j != depth - 1:
                # print('input SCAB:', feature_dims, feature_dims)
                self.down_path.append(SCAB(feature_dims, feature_dims))
            prev_channels = feature_dims

        self.up_path = nn.ModuleList()
        reverted_filters = filters[:-1][::-1]    # [512, 256, 128, 64]
        for j, feature_dims in enumerate(reverted_filters):
            # print('input UNetUpBlock:', feature_dims, j+2)
            self.up_path.append(UNetUpBlock(prev_channels, feature_dims, j+2))
            prev_channels = feature_dims

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # print(self.down_path, self.up_path, self.last)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if (i != len(self.down_path)-1) and (i % 2 == 1):
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)
    

class SigmaNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3, num_filter=64):
        super().__init__()
        block = []
        block.append(nn.Conv2d(in_channels, num_filter, kernel_size=3, padding=1, bias=True))
        block.append(nn.PReLU())

        for i in range(depth):
            block.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, bias=True))
            block.append(nn.PReLU())

        block.append(nn.Conv2d(num_filter, out_channels, kernel_size=3, padding=1, bias=True))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
    

class FAN2D(nn.Module):
    
    """ 
        Frequency Attention Network in 2D
        
        Funky implementation of UNET using attention blocks with 2D input 
        and working in the frequency domain via the wavelet transform.
        Activation functions throughout are PReLU and no BN, no dropout.
        This is a Black & White implementation of https://github.com/momo1689/FAN 
        
        Must set 1 dimension of 'patch_size' = 1 in config file: patch_size: [128, 1, 128]
        
    """
    
    def __init__(self, 
                 in_channels, 
                 g_filters=[64, 128, 256, 512, 1024], 
                 depth_S=5,
                 filters_S=64,
                 wave_pattern='db1', 
                 level=1, **kwargs):
        
        super().__init__()
        # in_channels should be 1 but ultimately it is defined by the level of wavelet decomposition
        assert in_channels == 1
        self.sigma_net = SigmaNet(in_channels=1, out_channels=1, depth=depth_S, num_filter=filters_S)
        self.UNet = UNet(in_channels=level*3+3, out_channels=level*3+1, filters=g_filters)
        self.wave_pattern = wave_pattern
        self.level = level
        self.decompose = SWTForward(wave=self.wave_pattern, level=self.level)
        self.reconstrution = SWTInverse(wave=self.wave_pattern, level=self.level)
        
        # self.out = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0),
        #     nn.ReLU()
        # )
        
    def forward(self, img):
        # SQUEEZE TO 2D
        for idx in range(2, 5):
            if img.size()[idx] == 1:
                img = img.squeeze(dim=idx)
                break
        
        noise_map = self.sigma_net(img)
        # Wavelet decomposition on gray scale image only. out = (cA, (cH, cV, cD))
        # Approximation coefficients (bs, 1, dim1, dim2)
        # Detail coefficients (bs, 3, dim1, dim2)
        cA, high_coeffs = self.decompose(img)
        # concat (Wavelet decomposition + image + noise_map) as input to UNET 
        # (bs, 6, dim1, dim2)
        net_input = torch.cat((cA, torch.cat(high_coeffs, dim=1), img, noise_map), dim=1) # dim=1 is color_channel

        net_out = self.UNet(net_input)

        out_cA = net_out[:, 0].unsqueeze(dim=1)
        out_coeffs = net_out[:, 1:self.level*3+1]
        # out_img = net_out[:, -1].unsqueeze(dim=1)
        out_recons = (out_cA, out_coeffs)
        out_wave = self.reconstrution(out_recons)

        # as an option, consider both direct and wavelet image together
        # out_combined = torch.cat((out_img, out_wave), dim=1)
        # out = self.out(out_combined).unsqueeze(dim=idx)
        
        # UNSQUEEZE to 3D
        out = out_wave.unsqueeze(dim=idx)
        return out
    
