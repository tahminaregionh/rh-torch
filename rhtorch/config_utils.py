#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ruamel.yaml as yaml
from datetime import datetime
from pathlib import Path
import torch
from rhtorch.version import __version__

loss_map = {'MeanAbsoluteError': 'mae',
            'MeanSquaredError': 'mse',
            'huber_loss': 'huber',
            'BCEWithLogitsLoss': 'BCE'}


def load_model_config(rootdir, arguments):
    
    # check for config_file
    config_file = Path(arguments.config)
    if not config_file.exists():
        config_file = rootdir.joinpath(config_file)
    if not config_file.exists():
        raise FileNotFoundError("Config file not found. Define relative to project directory or as absolute path in config file")
    
    # read the config file
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.RoundTripLoader)
        
    data_shape = 'x'.join(map(str, config['data_shape']))
    base_name = f"{config['module']}_{config['version_name']}_{config['data_generator']}"
    dat_name = f"bz{config['batch_size']}_{data_shape}"
    full_name = f"{base_name}_{dat_name}_k{arguments.kfold}_e{config['epoch']}"
    
    # check for data folder
    data_folder = Path(config['data_folder'])
    if not data_folder.exists():
        # try relative to project dir - in this case overwrite config
        data_folder = rootdir.joinpath(config['data_folder'])
    if not data_folder.exists():
        raise FileNotFoundError("Data path not found. Define relative to the project directory or as absolute path in config file")
    
    # additional info from args and miscellaneous to save in config
    config['build_date'] = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    config['model_name'] = full_name
    config['project_dir'] = str(rootdir)
    config['data_folder'] = str(data_folder)
    config['config_file'] = str(config_file)
    config['k_fold'] = arguments.kfold
    config['precision'] = arguments.precision
    config['GPUs'] = torch.cuda.device_count()
    config['global_batch_size'] = config['batch_size'] * config['GPUs']
    config['rhtorch_version'] = __version__
    if 'acc_grad_batches' not in config:
        config['acc_grad_batches'] = 1

    return config


def copy_model_config(path, config, append_timestamp=False):
    model_name = config['model_name']
    timestamp = config['build_date'].replace(' ','_')
    config_file = path.joinpath(f"config_{model_name}.yaml") if not append_timestamp else path.joinpath(f"config_{model_name}_{timestamp}.yaml")
    config.yaml_set_start_comment(f'Config file for {model_name}')
    with open(config_file, 'w') as file:
        yaml.dump(config, file, Dumper=yaml.RoundTripDumper)
