import yaml as yaml_default
import ruamel.yaml as yaml
from datetime import datetime
from pathlib import Path
import torch
from rhtorch.version import __version__
import socket
import torchio as tio
import pytorch_lightning as pl


class UserConfig:
    def __init__(self, arguments=None, mode='train'):

        self.config_file = self.is_path(arguments.config)
        self.rootdir = self.config_file.parent
        self.args = arguments

        # load user config file
        with open(self.config_file) as cf:
            self.hparams = yaml.load(cf, Loader=yaml.RoundTripLoader)

        # Fix when saved with default yaml instead
        if all([k in list(self.hparams.keys()) for k in ['state','dictitems']]):
            with open(self.config_file) as cf:
                self.hparams = yaml_default.load(cf, Loader=yaml_default.Loader)

        # for inference, only load the config file
        if mode == 'train':
            self.training_setup()

    def training_setup(self):
        # load default configs
        default_config_file = Path(__file__).parent.joinpath('default.config')
        with open(default_config_file) as dcf:
            self.default_params = yaml.load(dcf, Loader=yaml.Loader)

        # overwrite any parameters passed in throuch CLI
        self.cli_hparams()

        # merge the two dicts
        self.merge_dicts()

        # sanity check on data_folder provided by user
        self.data_path = self.is_path(self.hparams['data_folder'])

        if 'build_date' not in self.hparams:
            self.fill_additional_info()
        if 'model_name' not in self.hparams:
            # make model name
            self.create_model_name()

    def is_path(self, path):
        # check for path - assuming absolute path was given
        filepath = Path(path)
        if not filepath.exists():
            # assuming path was given relative to rootdir
            filepath = self.rootdir.joinpath(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"{path} not found. Define relative to project directory or as \
                  absolute path in config file/argument passing.")

        return filepath

    def merge_dicts(self):
        """ adds to the user_params dictionnary any missing key from the
            default params """

        for key, value in self.default_params.items():
            # copy from default if value is not None and key not
            # already in user config
            if value is not None and key not in self.hparams:
                self.hparams[key] = value

    def cli_hparams(self):
        # I don't know if that is the right way to go
        # (adding every key one by one)
        if self.args.learningrate:
            self.hparams['g_lr'] = self.args.learningrate
        if self.args.optimizer:
            self.hparams['g_optimizer'] = self.args.optimizer
        if self.args.activation:
            self.hparams['g_activation'] = self.args.activation
        if self.args.poolingtype:
            self.hparams['g_pooling_type'] = self.args.poolingtype

    def fill_additional_info(self):
        # additional info from args and miscellaneous to save in config
        self.hparams['build_date'] = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.hparams['color_channels_in'] = len(
            self.hparams['input_files']['name'])
        self.hparams['data_shape_in'] = [
            self.hparams['color_channels_in'], *self.hparams['patch_size']]
        self.hparams['project_dir'] = str(self.rootdir)
        self.hparams['data_folder'] = str(self.data_path)
        self.hparams['config_file'] = str(self.config_file)
        self.hparams['k_fold'] = self.args.kfold
        self.hparams['hostname'] = socket.gethostname()
        self.hparams['GPUs'] = torch.cuda.device_count()
        self.hparams['global_batch_size'] = self.hparams['batch_size'] * \
            self.hparams['GPUs']
        self.hparams['rhtorch_version'] = str(__version__)
        self.hparams['pytorch_version'] = str(torch.__version__)
        self.hparams['torchio_version'] = str(tio.__version__)
        self.hparams['pytorch_lightning_version'] = str(pl.__version__)

    def create_model_name(self):
        patch_size = 'x'.join(map(str, self.hparams['patch_size']))
        base_name = f"{self.hparams['module']}_{self.hparams['version_name']}_{self.hparams['data_generator']}"
        dat_name = f"bz{self.hparams['batch_size']}_{patch_size}"
        self.hparams['model_name'] = f"{base_name}_{dat_name}_k{self.args.kfold}_e{self.hparams['epoch']}"

    def save_copy(self, output_dir, append_timestamp=False):
        model_name = self.hparams['model_name']
        timestamp = f"_{self.hparams['build_date']}" if append_timestamp else ""
        save_config_file_name = f"config_{model_name}{timestamp}"
        config_file = output_dir.joinpath(save_config_file_name + ".yaml")
        self.hparams.yaml_set_start_comment(f'Config file for {model_name}')
        with open(config_file, 'w') as file:
            try:
                yaml.dump(self.hparams, file, Dumper=yaml.RoundTripDumper)
            except yaml.representer.RepresenterError:
                yaml_default.dump(self.hparams, file, default_flow_style=False)


    def pprint(self):
        print("\n####################### USER CONFIGS #######################")
        for k, v in self.hparams.items():

            if isinstance(v, yaml.comments.CommentedMap):
                print(k.ljust(40))
                for kp, vp in v.items():
                    print(' -', kp.ljust(37), vp)
            else:
                print(k.ljust(40), v)
        print("############################################################\n")
