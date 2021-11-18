import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
import numpy as np


class RegressConvNet(nn.Module):

    """
    Simple regression network consisting of series of conv_blocks
    (Conv, BatchNorm, Relu, Dropout) followed by one or more dense layers.
    Downsampling using max pooling between each conv_block except for the last.

    Number of blocks (conv and dense) can be scaled dynamically.

    Currently, this network can only be used to predict a single scalar value.
    The network can be used for regression and single-class classification.
    """

    def __init__(self, in_channels=1, convsizes=[7,7,5,3],
                 filters=[32,64,128,256], dropouts=[.1, .1, .1, .1],
                 dense_layers=[(None,1)], **kwargs):
        super().__init__()

        # Sanity check
        for d in dense_layers:
            assert d[0] is not None, \
                "OBS: Input number of features must be set based on data"

        self.chs = (in_channels, *filters)
        self.depth = len(filters)
        self.num_dense = len(dense_layers)
        self.conv_blocks = nn.ModuleList(
            [self.conv_block(
                self.chs[i],
                self.chs[i+1],
                convsizes[i],
                dropouts[i],
                do_pool=i<self.depth-1) \
            for i in range(self.depth)]
        )
        self.dense_blocks = nn.ModuleList(
            [self.dense_block(in_c=l[0], out_c=l[1]) for l in dense_layers]
        )

    def conv_block(self, in_c, out_c, convsize, drop_fraction, do_pool=True):
        modules = []
        modules.append(nn.Conv3d(in_c, out_c, kernel_size=convsize, stride=1))
        modules.append(nn.BatchNorm3d(out_c))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(drop_fraction))
        if do_pool:
            modules.append(nn.MaxPool3d(kernel_size=2, stride=2))
        return nn.Sequential(*modules)

    def dense_block(self, in_c, out_c):
        dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_c, out_c)
        )
        return dense

    def forward(self, x):
        for conv in self.conv_blocks:
            x = conv(x)
        for dense in self.dense_blocks:
            x = dense(x)
        return x
