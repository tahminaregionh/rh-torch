#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import os, sys
from pathlib import Path
# library package imports
from rhtorch.models import modules
from rhtorch.callbacks import plotting
from rhtorch.config_utils import load_model_config, copy_model_config

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Runs training on dataset in the input directory and config.yaml')
    parser.add_argument("-i", "--input", 
                        help="Directory containing data to train on. Parent folder contains 'config.yaml' file. Will use current working directory if nothing passed", 
                        type=str, default=os.getcwd())
    parser.add_argument("-c", "--config", help="Config file else than {project_name}_config.yaml in input folder", type=str, default='')
    parser.add_argument("-k", "--kfold", help="K-value for selecting train/test split subset. Default k=0", type=int, default=0)
    parser.add_argument("-t", "--test", help="Test run for 1 patient", action="store_true", default=False)
    parser.add_argument("--precision", help="Torch precision. Default 32", type=int, default=32)
    args = parser.parse_args()

    if args.test:
        os.environ['WANDB_MODE'] = 'dryrun'
        
    project_dir = Path(args.input)
    is_test = args.test
    
    # load configs and create additional entries for saving later
    configs, model_outname = load_model_config(project_dir, Path(args.config))
    if is_test:
        print('This is a test run on 10/2 train/test patients and 5 epochs.')
        configs['epoch'] = 5
    model_outname += f"_k{args.kfold}_e{configs['epoch']}"
    configs['project_dir'] = str(project_dir)
    configs['model_name'] = model_outname
    configs['k_fold'] = args.kfold
    configs['config_file'] = args.config
    
    # Set local data_generator
    sys.path.insert(1, args.input)
    import data_generator
    loader_params = {'batch_size': configs['batch_size'], 'num_workers': 4}
    data_gen = getattr(data_generator, configs['data_generator'])
    
    # training data
    augment = False if not 'augment' in configs.keys() else configs['augment']
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
    if 'pretrained_generator' in configs:
        assert configs['pretrained_generator'].endswith('.pt'), "When transfer learning, you must do it from a .pt file.\nTL from checkpoints is not yet implemented."
        model.load_state_dict(torch.load(configs['pretrained_generator']), strict=False)
        
    # wandb dashboard setup
    wandb_logger = WandbLogger(name=configs['version_name'],
                                project=configs['project_name'], 
                                log_model=False,
                                save_dir=project_dir,
                                config=configs)
    
    # CALLBACKS
    callbacks = []
    if 'callback_image2image' in list(configs.keys()):
        data_plot = data_gen('valid', conf=configs, augment=False, test=is_test)
        plot_dataloader = DataLoader(data_plot, **loader_params)
        callback_image2image = getattr(plotting, configs['callback_image2image'])
        callbacks.append( callback_image2image(plot_dataloader) )
        
    # checkpointing
    model_path = project_dir.joinpath('trained_models').joinpath(model_outname)
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
    callbacks.append( checkpoint_callback )
    
    # set the trainer and fit
    trainer = pl.Trainer(max_epochs=configs['epoch'], 
                         logger=wandb_logger, 
                         callbacks=callbacks, 
                         gpus=-1, 
                         accelerator='ddp',
                         resume_from_checkpoint=existing_checkpoint,
                         # stochastic_weight_avg=True,    # smooth the loss landscape to avoid local minimum
                         auto_select_gpus=True,    #)
                         auto_scale_batch_size='binsearch',
                         precision=args.precision,
                         profiler="simple")
    
    
    # actual training
    trainer.fit(model, train_dataloader, valid_dataloader)
    
    # save the model
    output_file = model_path.joinpath(f"{model_outname}.pt")
    torch.save(model.state_dict(), output_file)
    copy_model_config(model_path, configs)
    print("Saved model and config file to disk")
    
if __name__ == "__main__":
    main()