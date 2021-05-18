#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ruamel.yaml as yaml
from datetime import datetime
import torch

loss_map = {'MeanAbsoluteError': 'mae',
            'MeanSquaredError': 'mse',
            'huber_loss': 'huber',
            'BCEWithLogitsLoss': 'BCE'}


def load_model_config(path, cfg_file='config.yaml'):
    config_file = path.joinpath(cfg_file)
    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.RoundTripLoader)
        
    batch_size = cfg['batch_size'] * torch.cuda.device_count()

    base_name = f"{cfg['module']}_{cfg['version_name']}_{cfg['data_generator']}"
    dat_name = "_bz{}_{}".format(batch_size, 'x'.join(map(str, cfg['data_shape'])))
    name = base_name + dat_name
    cfg['build date'] = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    return cfg, name


def copy_model_config(path, config):
    model_name = config['model_name']
    config_file = path.joinpath(f"config_{model_name}.yaml")
    config.yaml_set_start_comment(f'Config file for {model_name}')
    with open(config_file, 'w') as file:
        yaml.dump(config, file, Dumper=yaml.RoundTripDumper)
