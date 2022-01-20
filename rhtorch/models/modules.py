# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
import math
from rhtorch.utilities.modules import recursive_find_python_class
import torchio as tio
import monai


class BaseModule(pl.LightningModule):
    """
    Generic Base module to start from, setting the default functionality.
    Note: Network is set in self.net (compared to generator etc in other modules)
    """
    def __init__(self, hparams, in_shape=None):
        super().__init__()
        try:
            self.hparams = hparams
        except AttributeError:
            self.hparams.update(hparams)
        # (self.img_rows, self.img_cols, self.channels_input)
        self.in_shape = in_shape
        self.in_channels, self.dimx, self.dimy, self.dimz = self.in_shape

        self.setup_model()

    def setup_model(self):
        self.net = recursive_find_python_class(self.hparams['net'])(
            self.in_channels, **self.hparams)
        self.optimizer = getattr(torch.optim, self.hparams['optimizer'])
        self.lr = self.hparams['lr']
        self.loss_train = getattr(tm, self.hparams['loss'])()
        self.loss_val = getattr(tm, self.hparams['loss'])()
        self.params = self.net.parameters()

    def forward(self, image):
        return self.net(image)

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
        x, y = self.prepare_batch(batch)
        y_hat = self.forward(x)
        loss = self.loss_train(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = self.prepare_batch(val_batch)
        y_hat = self.forward(x)
        loss = self.loss_val(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.params, lr=self.lr)

        if 'lr_scheduler' not in self.hparams:
            return optimizer
        else:
            print("LR_SCHEDULER:", self.hparams['lr_scheduler'])
            if self.hparams['lr_scheduler'] == 'polynomial_0.995':
                def lambda1(epoch): return 0.995 ** epoch
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda1)
            elif self.hparams['lr_scheduler'] == 'step_decay_0.8_25':
                drop = 0.8
                epochs_drop = 25.0
                def lambda1(epoch): return math.pow(
                    drop, math.floor((1+epoch)/epochs_drop))
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda1)
            elif self.hparams['lr_scheduler'] == 'exponential_decay_0.01':
                def lambda1(epoch): return math.exp(-0.01*epoch)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda1)
            else:
                print("MISSING SCHEDULER")
                exit(-1)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': self.hparams['lr_scheduler']
            }
            return [optimizer], [lr_scheduler]


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
        self.g_weight_decay = hparams['g_weight_decay'] if 'g_weight_decay' in self.hparams else 0
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
        g_optimizer = self.g_optimizer(self.g_params,
                                       lr=self.lr,
                                       weight_decay=self.g_weight_decay)

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
            elif self.hparams['lr_scheduler'] == 'step_decay_0.9_10':
                scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, 10, gamma=0.9)
            else:
                print("MISSING SCHEDULER")
                exit(-1)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': self.hparams['lr_scheduler']
            }
            return [g_optimizer], [lr_scheduler]


class LightningRAE(LightningAE):
    def __init__(self, hparams, in_shape=(2, 128, 128, 128)):
        super().__init__(hparams, in_shape)

    def forward(self, image):
        # return image - noise (generated by network)
        return image - self.generator(image)

    def training_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        # transform target image to noise image
        y = x - y
        # use generator instead of forward here to predict noise
        y_hat = self.generator(x)
        loss = self.g_loss_train(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_mse', self.mse_loss(y_hat, y))

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = self.prepare_batch(val_batch)
        y_hat = self.forward(x)
        loss = self.g_loss_val(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_mse', self.mse_loss(y_hat, y))

        return loss


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
        self.d_weight_decay = self.hparams['d_weight_decay'] if 'd_weight_decay' in self.hparams else 0
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
                           device=self.device, requires_grad=True)
        fake = torch.zeros((bs,) + self.disc_patch,
                           device=self.device, requires_grad=True)

        # losses to log only (not for training)
        self.log('train_mse', self.mse_loss(fake_imgs, tar))

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
            self.log('train_loss', g_loss)
            self.log('train_gan_loss', gan_loss)

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
        d_optimizer = self.d_optimizer(self.d_params,
                                       lr=self.d_lr,
                                       weight_decay=self.d_weight_decay)
        g_optimizer = self.g_optimizer(self.g_params,
                                       lr=self.lr,
                                       weight_decay=self.g_weight_decay)

        return [g_optimizer, d_optimizer]


class LightningRegressor(pl.LightningModule):
    """
        Module to perform regression on single output values rather
        than images. Can be used for single-class classification or regression.
        When is_classifier is set, output is sigmoid-activated for classification
        When 'dense_layers' is None, only one dense layer is added.
                            is a list, e.g. [256,1], two dense layers are added.
    """

    def __init__(self, hparams, in_shape=(1, 256, 256, 256)):
        super().__init__()
        try:
            self.hparams = hparams
        except AttributeError:
            self.hparams.update(hparams)

        self.in_shape = in_shape
        self.in_channels, self.dimx, self.dimy, self.dimz = self.in_shape

        # regressor
        # Calculate features used for dense connections
        num_dense_in_features = self.calculate_dense_features(
            img_size=self.in_shape[1:],
            filters=self.hparams['filters'],
            convolutions=self.hparams['convsizes'])
        if 'dense_layers' not in self.hparams:
            hparams['dense_layers'] = [(num_dense_in_features, 1)]
        elif all(isinstance(elem, list) for elem in self.hparams['dense_layers']):
            # For inference, this argument has already been formatted
            pass
        else:
            assert self.hparams['dense_layers'][-1] == 1, \
                ("The dense_layers argument must be a list ending with one "
                 "output feature in this regression module")
            dense_layers = []
            dense_features = [num_dense_in_features,
                              *self.hparams['dense_layers']]
            for i in range(len(dense_features)-1):
                dense_layers.append((dense_features[i], dense_features[i+1]))
            hparams['dense_layers'] = dense_layers  # Update hparams

        self.regressor = recursive_find_python_class(
            hparams['regressor'])(self.in_channels, **hparams)
        self.r_optimizer = getattr(torch.optim, hparams['r_optimizer'])
        self.lr = hparams['r_lr']
        if hparams['r_loss'] == "BCEWithLogitsLoss":
            self.r_loss_train = torch.nn.BCEWithLogitsLoss()
            self.r_loss_val = torch.nn.BCEWithLogitsLoss()
        else:
            self.r_loss_train = getattr(tm, hparams['r_loss'])()
            self.r_loss_val = getattr(tm, hparams['r_loss'])()
        self.r_params = self.regressor.parameters()

        # Diverge depending on module type: classification vs regression
        self.is_classifier = 'is_classifier' in self.hparams and \
                             self.hparams['is_classifier']
        if self.is_classifier:
            self.accuracy = tm.Accuracy()
            self.precision_ = tm.Precision(num_classes=1)
            self.recall = tm.Recall(num_classes=1)
            self.f1 = tm.F1(num_classes=1)
        else:
            pass  # More metrics could be added here

    def forward(self, image):
        return self.regressor(image)

    def calculate_dense_features(self, img_size, filters, convolutions):
        import functools
        conv_block = lambda s_, c_ : (s_-c_)+1
        max_pool = lambda s_ : s_//2
        for IDX, (f, c) in enumerate(zip(filters, convolutions)):
            img_size = list(map(functools.partial(conv_block, c_=c), img_size))
            if IDX < len(convolutions)-1:
                img_size = list(map(max_pool, img_size))
        count = img_size[0]*img_size[1]*img_size[2]*f
        return count

    def prepare_batch(self, batch):
        # necessary distinction for use of TORCHIO
        if isinstance(batch, dict):
            # first input channel
            x = batch['input0'][tio.DATA]

            # other input channels if any
            for i in range(1, self.in_channels):
                x_i = batch[f'input{i}'][tio.DATA]
                color_channel_axis = 0 if x.ndim==4 else 1 # except 2D..
                x = torch.cat((x, x_i), axis=color_channel_axis)

            y = batch['target0'] if torch.is_tensor(batch['target0']) else torch.tensor([batch['target0']])
            y = y.unsqueeze(-1).float()

            return x, y

        # normal use case
        else:
            return batch

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = self.prepare_batch(batch)
        y_hat = self.forward(x)
        # main loss used for optimization
        loss = self.r_loss_train(y_hat, y)
        self.log('train_loss', loss)

        # Other metrics
        if self.is_classifier:
            y_pred = torch.sigmoid(y_hat)
            y_int = y.int()
            self.log('train_accuracy', self.accuracy(y_pred, y_int))
            self.log('train_precision_', self.precision_(y_pred, y_int))
            self.log('train_recall', self.recall(y_pred, y_int))
            self.log('train_f1', self.f1(y_pred, y_int))
        else:
            pass

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = self.prepare_batch(val_batch)
        y_hat = self.forward(x)
        loss = self.r_loss_val(y_hat, y)
        self.log('val_loss', loss)

        # Other metrics
        if self.is_classifier:
            y_pred = torch.sigmoid(y_hat)
            y_int = y.int()
            self.log('val_accuracy', self.accuracy(y_pred, y_int))
            self.log('val_precision_', self.precision_(y_pred, y_int))
            self.log('val_recall', self.recall(y_pred, y_int))
            self.log('val_f1', self.f1(y_pred, y_int))
        else:
            pass

        return loss

    def configure_optimizers(self):
        r_optimizer = self.r_optimizer(self.r_params, lr=self.lr)

        if 'lr_scheduler' not in self.hparams:
            return r_optimizer
        else:
            print("LR_SCHEDULER:", self.hparams['lr_scheduler'])
            if self.hparams['lr_scheduler'] == 'polynomial_0.995':
                def lambda1(epoch): return 0.995 ** epoch
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    r_optimizer, lr_lambda=lambda1)
            elif self.hparams['lr_scheduler'] == 'step_decay_0.8_25':
                drop = 0.8
                epochs_drop = 25.0
                def lambda1(epoch): return math.pow(
                    drop, math.floor((1+epoch)/epochs_drop))
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    r_optimizer, lr_lambda=lambda1)
            elif self.hparams['lr_scheduler'] == 'exponential_decay_0.01':
                def lambda1(epoch): return math.exp(-0.01*epoch)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    r_optimizer, lr_lambda=lambda1)
            else:
                print("MISSING SCHEDULER")
                exit(-1)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': self.hparams['lr_scheduler']
            }
            return [r_optimizer], [lr_scheduler]


class MONAI(BaseModule):
    """
    Uses MONAI to define the network.
    Following shows use case for a DenseNet121. Config file must have, e.g.:
    >>>
    module: MONAI
    monai_params:
      net: DenseNet121
      args:
        spatial_dims: 3
        in_channels: 4
        out_channels: 1
    >>>
    where arguments under "args" is specific to the MONAI network.
    See monai documentation.
    """
    def __init__(self, hparams, in_shape=(1, 256, 256, 256)):
        super().__init__(hparams, in_shape)

    def setup_model(self):

        # Add MONAI network
        monai_params = self.hparams['monai_params']
        assert hasattr(monai.networks.nets, monai_params['net']),\
            "MONAI does not have network {}.".format(monai_params['net'])
        net = getattr(monai.networks.nets, monai_params['net'])
        self.net = net(**monai_params['args'])

        # Continue default for optimizer and loss
        self.optimizer = getattr(torch.optim, self.hparams['optimizer'])
        self.lr = self.hparams['lr']
        self.loss_train = getattr(tm, self.hparams['loss'])()
        self.loss_val = getattr(tm, self.hparams['loss'])()
        self.params = self.net.parameters()
