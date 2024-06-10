#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
import sys
from pathlib import Path
import argparse

# library package imports
import rhtorch
from rhtorch.callbacks import plotting
from rhtorch.utilities.config import UserConfig
from rhtorch.utilities.modules import recursive_find_python_class


def main():
    parser = argparse.ArgumentParser(
        description='Runs training on dataset in the input directory and config.yaml')
    parser.add_argument("-c", "--config",
                        help="Config file else than 'config.yaml' in project directory (input dir)",
                        type=str, default=os.getcwd() + '/config.yaml')
    parser.add_argument("-k", "--kfold",
                        help="K-value for selecting train/test split subset. Default k=0",
                        type=int, default=0)
    parser.add_argument("-t", "--test", help="Test run for 1 patient",
                        action="store_true", default=False)

    # args for wandb sweep - I don't know if this is the right way to go
    parser.add_argument("-lr", "--learningrate",
                        help="Learning rate of generator",
                        type=float, default=0)
    parser.add_argument("-opt", "--optimizer",
                        help="Optimizer for the generator",
                        type=str, default='')
    parser.add_argument("-act", "--activation",
                        help="Activation used throughout the generator architecture",
                        type=str, default='')
    parser.add_argument("-pool", "--poolingtype",
                        help="Down sampling layer type for UNet3D generator",
                        type=str, default='')

    args = parser.parse_args()
    user_configs = UserConfig(args)
    project_dir = user_configs.rootdir
    is_test = args.test

    # setting up test
    if is_test:
        print('This is a test run on 10/2 train/test patients and 5 epochs.')
        user_configs.hparams['epoch'] = 5
        user_configs.create_model_name()  # Update name using newly set epoch
        os.environ['WANDB_MODE'] = 'dryrun'
    configs = user_configs.hparams

    # Set local data_generator
    sys.path.insert(1, str(project_dir))
    import data_generator
    data_gen = getattr(data_generator, configs['data_generator'])
    data_module = data_gen(configs, quick_test=is_test)
    data_module.prepare_data()
    data_module.setup()
    print('Done preparing the data.')

    # Augmentation message
    if 'augment' in configs and configs['augment']:
        print("Augmenting data")

    module = recursive_find_python_class(configs['module'])
    # Should also be changed to custom arguments (**configs)
    shape_in = configs['data_shape_in']   # color_channel, dim1, dim2, dim3
    model = module(configs, shape_in)
    
    # transfer learning setup
    if 'pretrained_generator' in configs and configs['pretrained_generator']:
        tl_message = "Setting up transfer learning"
        pretrained_model_path = Path(configs['pretrained_generator'])
        if pretrained_model_path.exists():
            if pretrained_model_path.name.endswith(".ckpt"):
                # important to pass in new configs here as we want to load the weights but config may differ from pretrained model
                model = module.load_from_checkpoint(
                    pretrained_model_path, hparams=configs, in_shape=shape_in, strict=False)
            elif pretrained_model_path.name.endswith(".pt"):
                ckpt = torch.load(pretrained_model_path)
                pretrained_model = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                model.load_state_dict(pretrained_model, strict=False)
            else:
                raise ValueError("Expected model format: '.pt' or '.ckpt'.")
        else:
            raise FileNotFoundError(
                "Model file not found. Check your path in config file.")

        # This should be more generic. What if we are to only freeze some of the layers?
        if 'freeze_encoder' in configs and configs['freeze_encoder']:
            tl_message += "with frozen encoder."
            model.encoder.freeze()

        print(tl_message)

    # wandb dashboard setup
    wandb_logger = WandbLogger(name=configs['version_name'],
                               project=configs['project_name'],
                               log_model=False,
                               save_dir=project_dir,
                               config=configs)

    # callbacks
    callbacks = []
    if 'plotting_callback' in configs:
        plot_configs = configs['plotting_callback']
        plotting_callback = recursive_find_python_class(plot_configs['class'], current_module='rhtorch.callbacks')
        callbacks.append(plotting_callback(model, data_module, configs))

    # checkpointing
    model_path = project_dir.joinpath(
        'trained_models').joinpath(configs['model_name'])
    checkpoint_dir = model_path.joinpath('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoint = checkpoint_dir.joinpath('last.ckpt')
    existing_checkpoint = None \
        if not existing_checkpoint.exists() else str(existing_checkpoint)
    checkpoint_dir = str(checkpoint_dir)
    # str() added to fix fsspec error with pathlib.path.stat() missing
    # follow_symlinks argument

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='Checkpoint_min_val_loss-{epoch:03d}',
        save_top_k=3,       # saves 3 best models based on monitored value
        save_last=True,     # additionally overwrites a file last.ckpt after each epoch
        every_n_epochs=2,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if 'early_stopping_callback' in configs:
        early_stopping_config = configs['early_stopping_callback']
        early_stop_callback = EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=early_stopping_config['patience'])
        callbacks.append(early_stop_callback)

    # Save the config prior to training the model - one for each time the script is started
    if model.local_rank == 0:
        if not is_test:
            user_configs.save_copy(model_path, append_timestamp=True)
            print("Saved config prior to model training")
        user_configs.pprint()

    # set the trainer and fit
    strategy = 'ddp' if configs['GPUs'] > 1 else 'auto'
    trainer = pl.Trainer(max_epochs=configs['epoch'],
                         logger=wandb_logger,
                         callbacks=callbacks,
                         devices=-1,
                         strategy=strategy,
                         accumulate_grad_batches=configs['acc_grad_batches'],
                         precision=configs['precision'],
                         profiler="simple")

    # actual training    
    trainer.fit(model, datamodule=data_module, ckpt_path=existing_checkpoint)

    # add useful info to saved configs
    user_configs.hparams['best_model'] = checkpoint_callback.best_model_path

    # save the model
    output_file = model_path.joinpath(f"{configs['model_name']}.pt")
    torch.save(model.state_dict(), output_file)
    user_configs.save_copy(model_path)
    print("Saved model and config file to disk")


if __name__ == "__main__":
    main()
