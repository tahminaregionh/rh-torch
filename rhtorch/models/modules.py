# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
import math
from rhtorch.utilities.modules import recursive_find_python_class
import torchio as tio


class LightningAE(pl.LightningModule):
    def __init__(self, hparams, in_shape=(2, 128, 128, 128)):
        super().__init__()
        try:
            self.hparams = hparams
        except AttributeError:
            self.hparams.update(hparams)
        # (self.img_rows, self.img_cols, self.channels_input)
        self.in_shape = in_shape
        self.in_channels, self.dimx, self.dimy, self.dimz = self.in_shape

        # generator
        self.generator = recursive_find_python_class(
            hparams['generator'])(self.in_channels, **hparams)
        self.g_optimizer = getattr(torch.optim, hparams['g_optimizer'])
        self.lr = hparams['g_lr']
        self.g_loss_train = getattr(tm, hparams['g_loss'])()  # MAE
        self.g_loss_val = getattr(tm, hparams['g_loss'])()  # MAE
        self.g_params = self.generator.parameters()

        # additional losses
        self.mse_loss = tm.MeanSquaredError()

    def forward(self, image):
        """ image.size: (Batch size, Color channels, Depth, Height, Width) """
        return self.generator(image)

    def prepare_batch(self, batch):
        # necessary distinction for use of TORCHIO
        if isinstance(batch, dict):
            # first input channel
            x = batch['input0'][tio.DATA]
            # other input channels if any
            for i in range(1, self.in_channels):
                x_i = batch[f'input{i}'][tio.DATA]
                # axis=0 is batch_size, axis=1 is color_channel
                x = torch.cat((x, x_i), axis=1)
            # target channel
            y = batch['target0'][tio.DATA]
            return x, y

        # normal use case
        else:
            return batch

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = self.prepare_batch(batch)   # instead of x, y = batch
        y_hat = self.forward(x)
        # main loss used for optimization
        loss = self.g_loss_train(y_hat, y)
        # for single GPU training, sync_dist should be False
        # however we can drop that key altogether when using torchmetrics
        # see: https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#synchronize-validation-and-test-logging
        self.log('train_loss', loss)  # , sync_dist=True)
        # other losses to log only
        self.log('train_mse', self.mse_loss(y_hat, y))  # , sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = self.prepare_batch(val_batch)    # instead of x, y = val_batch
        y_hat = self.forward(x)
        loss = self.g_loss_val(y_hat, y)
        self.log('val_loss', loss)  # , sync_dist=True)
        self.log('val_mse', self.mse_loss(y_hat, y))  # , sync_dist=True)

        return loss

    def configure_optimizers(self):
        g_optimizer = self.g_optimizer(self.g_params, lr=self.lr)

        if 'lr_scheduler' not in self.hparams:
            return g_optimizer
        else:
            print("LR_SCHEDULER:", self.hparams['lr_scheduler'])
            if self.hparams['lr_scheduler'] == 'polynomial_0.995':
                def lambda1(epoch): return 0.995 ** epoch
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    g_optimizer, lr_lambda=lambda1)
            elif self.hparams['lr_scheduler'] == 'step_decay_0.8_25':
                drop = 0.8
                epochs_drop = 25.0
                def lambda1(epoch): return math.pow(
                    drop, math.floor((1+epoch)/epochs_drop))
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    g_optimizer, lr_lambda=lambda1)
            elif self.hparams['lr_scheduler'] == 'exponential_decay_0.01':
                def lambda1(epoch): return math.exp(-0.01*epoch)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    g_optimizer, lr_lambda=lambda1)
            else:
                print("MISSING SCHEDULER")
                exit(-1)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': self.hparams['lr_scheduler']
            }
            return [g_optimizer], [lr_scheduler]


class LightningPix2Pix(LightningAE):

    def __init__(self, hparams, in_shape=(2, 128, 128, 128)):
        super().__init__(hparams, in_shape)

        # Calculate output shape of D (PatchGAN) (4x halving for the 4 Conv3D layers)
        patchx = int(self.dimx / 2**4)
        patchy = int(self.dimy / 2**4)
        patchz = int(self.dimz / 2**4)
        self.disc_patch = (1, patchx, patchy, patchz)

        # discriminator
        self.discriminator = recursive_find_python_class(
            hparams['discriminator'])(self.in_channels)
        self.d_optimizer = getattr(torch.optim, self.hparams['d_optimizer'])
        self.d_lr = self.hparams['d_lr']
        # BCE with logits
        self.d_loss = getattr(nn, self.hparams['d_loss'])()
        self.d_params = self.discriminator.parameters()
        self.LAMBDA = 100

    def toggle_optimizer(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defines the training loop. It is independent of forward
        inp, tar = self.prepare_batch(batch)
        fake_imgs = self.generator(inp)
        bs = inp.size()[0]

        # (bs, 1, 16/16=1, 128/16=8, 128/16=8)    # 16=2**4 for 4x down conv of the patch
        valid = torch.ones((bs,) + self.disc_patch,
                           device=self.device, requires_grad=False)
        fake = torch.zeros((bs,) + self.disc_patch,
                           device=self.device, requires_grad=False)

        # losses to log only (not for training)-> raise problems
        # self.log('train_accuracy', self.accuracy(fake_imgs, tar))
        self.log('train_mse', self.mse_loss(fake_imgs, tar)) #, sync_dist=True)

        #  Train Generator through the GAN
        if optimizer_idx == 0:
            # No need to calculate the gradients for Discriminators' parameters when training the generator
            for param in self.discriminator.parameters():
                param.requires_grad = False

            patches_out = self.discriminator(inp, fake_imgs)
            # telling the discriminator that generated images are real
            d_loss = self.d_loss(patches_out, valid)
            # mean absolute error
            g_loss = self.g_loss_train(fake_imgs, tar)
            gan_loss = d_loss + (self.LAMBDA * g_loss)
            # implied for generator
            self.log('train_loss', g_loss) #, sync_dist=True)
            self.log('train_gan_loss', gan_loss) #, sync_dist=True)

            return gan_loss

        #  Train Discriminator
        if optimizer_idx == 1:
            # reset calculation of gradient to True
            for param in self.discriminator.parameters():
                param.requires_grad = True

            # train discriminator with real images
            real_patch = self.discriminator(inp, tar)
            d_loss_real = self.d_loss(real_patch, valid)
            # train discriminator with fake (generated) images
            fake_patch = self.discriminator(inp, fake_imgs)
            d_loss_fake = self.d_loss(fake_patch, fake)
            # combine the losses equally
            d_loss = 0.5 * torch.add(d_loss_real, d_loss_fake)
            self.log('train_d_loss', d_loss, sync_dist=True)

            return d_loss

    def configure_optimizers(self):
        # can return multiple optimizers in a tuple, for GANs for instance
        d_optimizer = self.d_optimizer(self.d_params, lr=self.d_lr)
        g_optimizer = self.g_optimizer(self.g_params, lr=self.lr)

        return [g_optimizer, d_optimizer]
