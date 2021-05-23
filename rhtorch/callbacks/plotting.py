from pytorch_lightning.callbacks import Callback
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_inline(d1, d2, d3, color_channel_axis=0, vmin=None, vmax=None):
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
        d1 = d1[0,...] if color_channel_axis == 0 else d1[...,0]
        d1 = torch.unsqueeze(d1, color_channel_axis)
    d_arr = np.concatenate((d1, d2, d3), color_channel_axis)
    num_dat = d_arr.shape[color_channel_axis]
    
    fig, ax = plt.subplots(1, num_dat, gridspec_kw={'wspace': 0, 'hspace': 0})
    slice_i = int(d1.size(1) / 2)
    orient = 0
    text_pos = d1.size(2) * 0.98
    
    titles = ['Input', 'Target', 'Prediction']
    
    for idx in range(num_dat):
        single_data = d_arr.take(indices=idx, axis=color_channel_axis) 
        ax[idx].imshow(single_data.take(indices=slice_i, axis=orient), 
                       cmap='gray', vmin=vmin, vmax=vmax)
        ax[idx].axis('off')
        ax[idx].text(3, text_pos, titles[idx], color='white', fontsize=12)
        
    fig.tight_layout()
    wandb_im = wandb.Image(fig)
    plt.close()
    return wandb_im

class ImagePredictionLogger(Callback):
    def __init__(self, val_dataloader, config=None):
        super().__init__()
        self.X, self.y = next(iter(val_dataloader))
        
        # TODO: Read config file to parse vmin, vmax or e.g. custom titles
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Dataloader loads on CPU --> pass to GPU
        X = self.X.to(device=pl_module.device)
        y_hat = pl_module(X)
        
        # move arrays back to the CPU for plotting
        X = X.cpu()
        y = self.y
        y_hat = y_hat.cpu()
        
        # generate figures in a list
        figs = [plot_inline(im1, im2, im3) for im1, im2, im3 in zip(X, y, y_hat)]
        
        # add to logger like so
        trainer.logger.experiment.log({"Sample images": figs})

