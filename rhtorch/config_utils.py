import ruamel.yaml as yaml
from datetime import datetime
from pathlib import Path
import torch
from rhtorch.version import __version__
import socket

class UserConfig:
    def __init__(self, rootdir, arguments=None):
        self.rootdir = rootdir
        self.config_file = self.is_path(arguments.config)
        self.args = arguments
        
        # load default configs
        default_config_file = Path(__file__).parent.joinpath('default_config.yaml')
        with open(default_config_file) as dcf:
            self.default_params = yaml.load(dcf, Loader=yaml.Loader)
            
        # load user config file
        with open(self.config_file) as cf:
            self.hparams = yaml.load(cf, Loader=yaml.RoundTripLoader)
        
        # merge the two dicts
        self.merge_dicts()
        
        # sanity check on data_folder provided by user
        self.data_path = self.is_path(self.hparams['data_folder'])
        
        # check whether distributed learning
        gpu_count = torch.cuda.device_count()
        self.hparams['gpu_count'] = gpu_count
        self.hparams['effective_batch_size'] = self.hparams['batch_size'] * gpu_count
        
        # make model name
        self.create_model_name()
        self.fill_additional_info()
            
    def is_path(self, path):
        # check for path - assuming absolute path was given
        filepath = Path(path)
        if not filepath.exists():
            # assuming path was given relative to rootdir
            filepath = self.rootdir.joinpath(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"{path} not found. Define relative to project directory or as absolute path in config file/argument passing.")
        
        return filepath

    def merge_dicts(self):
        """ adds to the user_params dictionnary any missing key from the default params """
        
        for k, v in self.default_params.items():
            if k not in self.hparams:
                self.hparams[k] = v
            ### TO DO - ENSURE NOT COPYING IRRELEVANT DATA e.g. GAN parameters if model is AE
    
    def create_model_name(self):
        
        data_shape = 'x'.join(map(str, self.hparams['data_shape']))
        base_name = f"{self.hparams['module']}_{self.hparams['version_name']}_{self.hparams['data_generator']}"
        dat_name = f"bz{self.hparams['effective_batch_size']}_{data_shape}"
        self.hparams['model_name'] = f"{base_name}_{dat_name}_k{self.args.kfold}_e{self.hparams['epoch']}"

    def fill_additional_info(self):
        # additional info from args and miscellaneous to save in config
        self.hparams['build date'] = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.hparams['model_name'] = self.hparams['model_name']
        self.hparams['project_dir'] = str(self.rootdir)
        self.hparams['data_folder'] = str(self.data_path)
        self.hparams['config_file'] = str(self.config_file)
        self.hparams['k_fold'] = self.args.kfold
        self.hparams['GPUs'] = torch.cuda.device_count()
        self.hparams['global_batch_size'] = self.hparams['batch_size'] * self.hparams['GPUs']
        self.hparams['rhtorch_version'] = __version__
        self.hparams['hostname'] = socket.gethostname()
    
    def save_copy(self, output_dir, append_timestamp=False):
        model_name = self.hparams['model_name']
        timestamp = f"_{self.hparams['build_date']}" if append_timestamp else ""
        save_config_file_name = f"config_{model_name}{timestamp}"        
        config_file = output_dir.joinpath(save_config_file_name + ".yaml")
        self.hparams.yaml_set_start_comment(f'Config file for {model_name}')
        with open(config_file, 'w') as file:
            yaml.dump(self.hparams, file, Dumper=yaml.RoundTripDumper)
