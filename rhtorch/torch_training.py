#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path

# library package imports
from rhtorch.models import modules
from rhtorch.callbacks import plotting
from rhtorch.config_utils import UserConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs training on dataset in the input directory and config.yaml')
    parser.add_argument("-i", "--input",
                        help="Project directory. Should contain data folder to train on + config file + data_generator. Will use current working directory if nothing passed",
                        type=str, default=os.getcwd())
    parser.add_argument("-c", "--config",
                        help="Config file else than 'config.yaml' in project directory (input dir)",
                        type=str, default='config.yaml')
    parser.add_argument("-k", "--kfold",
                        help="K-value for selecting train/test split subset. Default k=0",
                        type=int, default=0)
    parser.add_argument("-t", "--test", help="Test run for 1 patient",
                        action="store_true", default=False)

    args = parser.parse_args()
    project_dir = Path(args.input)
    is_test = args.test

    # load configs from file + additional info from args
    user_configs = UserConfig(project_dir, args)
    configs = user_configs.hparams

    # Set local data_generator
    sys.path.insert(1, args.input)
    import data_generator
    loader_params = {'batch_size': configs['batch_size'], 'num_workers': 4}
    data_gen = getattr(data_generator, configs['data_generator'])

    # setting up test
    if is_test:
        print('This is a test run on 10/2 train/test patients and 5 epochs.')
        configs['epoch'] = 5
        os.environ['WANDB_MODE'] = 'dryrun'

    # training data
    augment = False if 'augment' not in configs else configs['augment']
    if augment:
        print("Augmenting data")
    data_train = data_gen('train', conf=configs, augment=augment, test=is_test)
    train_dataloader = DataLoader(data_train, shuffle=True, **loader_params)

    # validation data
    data_valid = data_gen('valid', conf=configs, augment=False, test=is_test)
    valid_dataloader = DataLoader(data_valid, **loader_params)

    # define lightning module
    shape_in = data_train.data_shape_in
    module = getattr(modules, configs['module'])
    model = module(configs, shape_in)

    # transfer learning setup
    if 'pretrained generator' in configs and configs['pretrained_generator']:
        message = "Setting up transfer learning"
        pretrained_model_path = Path(configs['pretrained_generator'])
        if pretrained_model_path.exists():
            if pretrained_model_path.name.endswith(".ckpt"):
                # important to pass in new configs here as we want to load the weights but config may differ from pretrained model
                model = module.load_from_checkpoint(
                    pretrained_model_path, hparams=configs, in_shape=shape_in, strict=False)
            elif pretrained_model_path.name.endswith(".pt"):
                ckpt = torch.load(pretrained_model_path)
                # OBS, the 'state_dict' is not set during save?
                # What if we are to save multiple models used later for pretrain? (e.g. a GAN with 3 networks?)
                pretrained_model = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                model.load_state_dict(pretrained_model, strict=False)
            else:
                raise ValueError("Expected model format: '.pt' or '.ckpt'.")
        else:
            raise FileNotFoundError(
                "Model file not found. Check your path in config file.")

        if 'freeze_encoder' in configs and configs['freeze_encoder']:
            message += "with frozen encoder."
            model.encoder.freeze()

        print(message)

    # wandb dashboard setup
    wandb_logger = WandbLogger(name=configs['version_name'],
                               project=configs['project_name'],
                               log_model=False,
                               save_dir=project_dir,
                               config=configs)

    # callbacks
    callbacks = []
    if 'callback_image2image' in configs:
        data_plot = data_gen('valid', conf=configs,
                             augment=False, test=is_test)
        plot_dataloader = DataLoader(data_plot, **loader_params)
        callback_image2image = getattr(
            plotting, configs['callback_image2image'])
        callbacks.append(callback_image2image(plot_dataloader))

    # checkpointing
    model_path = project_dir.joinpath(
        'trained_models').joinpath(configs['model_name'])
    model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_path.joinpath('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoint = checkpoint_dir.joinpath('last.ckpt')
    if not existing_checkpoint.exists():
        existing_checkpoint = None

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=f'Checkpoint_min_val_loss',
        save_top_k=3,       # saves 3 best models based on monitored value
        save_last=True,     # additionally overwrites a file last.ckpt after each epoch
        period=2,
    )
    callbacks.append(checkpoint_callback)

    # Save the config prior to training the model - one for each time the script is started
    if not is_test:
        user_configs.save_copy(model_path, append_timestamp=True)
        print("Saved config prior to model training")

    # set the trainer and fit
    accelerator = 'ddp' if configs['GPUs'] > 1 else None
    trainer = pl.Trainer(max_epochs=configs['epoch'],
                         logger=wandb_logger,
                         callbacks=callbacks,
                         gpus=-1,
                         accelerator=accelerator,
                         resume_from_checkpoint=existing_checkpoint,
                         auto_select_gpus=True,
                         accumulate_grad_batches=configs['acc_grad_batches'],
                         precision=configs['precision'],
                         profiler="simple")

    # actual training
    trainer.fit(model, train_dataloader, valid_dataloader)

    # add useful info to saved configs
    user_configs.hparams['best_model'] = checkpoint_callback.best_model_path

    # save the model
    output_file = model_path.joinpath(f"{configs['model_name']}.pt")
    torch.save(model.state_dict(), output_file)
    user_configs.save_copy(model_path)
    print("Saved model and config file to disk")


if __name__ == "__main__":
    main()
