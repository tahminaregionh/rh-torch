from pytorch_lightning.callbacks import Callback
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio


class Image2ImageLogger(Callback):
    def __init__(self, data_module, config=None):
        super().__init__()
        val_data = data_module.val_dataloader()
        batch = next(iter(val_data))
        self.color_channels = config['color_channels_in']
        self.X, self.y = self.prepare_batch(batch)

        # plotting properties from config
        plot_configs = config['plotting_callback']
        self.viewing_axis = plot_configs['viewing_axis']
        # self.fixed_slice = config['fixed_slice']
        self.vmin = plot_configs['vmin']
        self.vmax = plot_configs['vmax']
        self.titles = ['Input', 'Target', 'Prediction']
    
    def prepare_batch(self, batch):
        # necessary distinction for use of TORCHIO
        if isinstance(batch, dict):
            # first input channel
            x = batch['input0'][tio.DATA]
            # other input channels if any
            for i in range(1, self.color_channels):
                x_i = batch[f'input{i}'][tio.DATA]
                # axis=0 is batch_size, axis=1 is color_channel
                x = torch.cat((x, x_i), axis=1)
            # target channel
            y = batch['target0'][tio.DATA]
            return x, y

        # normal use case
        else:
            return batch

    def plot_inline(self, d1, d2, d3, color_channel_axis=0):
        """
        Parameters
        ----------
        d1 : numpy.ndarray
            Input data to a model
        d2 : numpy.ndarray
            Ground truth data
        d3 : numpy.ndarray
            Infered data based on input data
        color_channel_axis : int, optional
            Axis for color channel in the numpy array . 
            Default is 0 for Pytorch models (cc, dimx, dimy, dimz)
            Use 3 for TF models (dimx, dimy, dimz, cc)
        vmin : Lower bound for color channel. Default (None) used to plot full range
        vmax : Upper bound for color channel. Default (None) used to plot full range

        """
        # If input has more than 1 color channel, use only the first
        if d1.shape[color_channel_axis] > 1:
            d1 = d1[0, ...] if color_channel_axis == 0 else d1[..., 0]
            d1 = torch.unsqueeze(d1, color_channel_axis)
        d_arr = np.concatenate((d1, d2, d3), color_channel_axis)
        num_dat = d_arr.shape[color_channel_axis]

        fig, ax = plt.subplots(1, num_dat, gridspec_kw={
                               'wspace': 0, 'hspace': 0})
        # slice_i = int(d1.size(self.viewing_axis) / 2)
        # text_pos = d1.size(2) * 0.98

        for idx in range(num_dat):
            single_data = d_arr.take(indices=idx, axis=color_channel_axis)
            slice_i = int(single_data.shape[self.viewing_axis] / 2)
            single_slice = single_data.take(indices=slice_i, axis=self.viewing_axis)
            text_pos = single_slice.shape[0] * 0.98
            ax[idx].imshow(single_slice, cmap='gray', vmin=self.vmin, vmax=self.vmax)
            ax[idx].axis('off')
            ax[idx].text(3, text_pos, self.titles[idx],
                         color='white', fontsize=12)

        fig.tight_layout()
        wandb_im = wandb.Image(fig)
        plt.close()
        return wandb_im

    def on_validation_epoch_end(self, trainer, pl_module):
        # Dataloader loads on CPU --> pass to GPU
        X = self.X.to(device=pl_module.device)
        y_hat = pl_module(X)

        # move arrays back to the CPU for plotting
        X = X.cpu()
        y = self.y
        y_hat = y_hat.cpu()

        # generate figures in a list
        figs = [self.plot_inline(im1, im2, im3)
                for im1, im2, im3 in zip(X, y, y_hat)]

        # add to logger like so
        trainer.logger.experiment.log({"Sample images": figs})
