from torch.utils.data import Dataset
from rhtorch.data.DataAugmentation3D import DataAugmentation3D
from pathlib import Path
import torch
import json
import pickle
import glob
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchio import ScalarImage, Subject, SubjectsDataset, Queue
from torchio.transforms import Lambda, RandomAffine, RandomFlip, Compose
from torchio.data import UniformSampler
from sklearn.model_selection import train_test_split

augmenter = DataAugmentation3D(rotation_range=[5, 5, 5],
                               shift_range=[0.05, 0.05, 0.05],
                               shear_range=[2, 2, 0],
                               zoom_lower=[0.9, 0.9, 0.9],
                               zoom_upper=[1.2, 1.2, 1.2],
                               zoom_independent=True,
                               data_format='channels_last',    # relative to position of batch_size
                               flip_axis=[0, 1, 2],
                               fill_mode='reflect')


def swap_axes(x):
    return np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)


def numpy_reader(path):
    return np.load(path), np.eye(4)


def load_data_splitting(mode="train", pkl_file=None, datadir=None, fold=0, duplicate_list=0, quick_test=False):

    # search for pkl/json file in data folder if none passed
    if pkl_file:
        pkl_file = datadir.joinpath(pkl_file)
    else:
        pkl_file = glob.glob(f"{datadir}/*_train_test_split_*_fold.json")[0]
    if not pkl_file.exists():
        raise FileNotFoundError(
            "Data train/test split info file not found. Add file to data folder or declare in config file.")

    # use json or pickle loader depending on file extension
    load_module = json if pkl_file.name.endswith(".json") else pickle
    with open(pkl_file, 'r') as f:
        split_data_info = load_module.load(f)

    patients = split_data_info[f"{mode}_{fold}"]
    if duplicate_list:
        patients = np.repeat(patients, duplicate_list)

    # trim the test/train set if quick test
    if quick_test:
        keep = 2 if mode == 'test' else 10
        patients = patients[:keep]

    return patients

## TORCHIO way of setting up data module

class TIODataModule(pl.LightningDataModule):
    def __init__(self, config, quick_test=False):
        super().__init__()
        self.config = config
        self.k = config['k_fold']
        self.quick_test = quick_test
        self.batch_size = self.config['batch_size']
        self.datadir = Path(self.config['data_folder'])
        self.augment = self.config['augment']
        self.num_workers = 4
        # normalization factor for PET data
        self.pet_norm = self.config['pet_normalization_constant']

        # for the queue
        self.patch_size = self.config['patch_size']  # [16, 128, 128]
        patches_per_volume = int(
            np.max(self.patch_size) / np.min(self.patch_size))
        self.queue_length = patches_per_volume
        self.samples_per_volume = int(
            patches_per_volume / 2) if patches_per_volume > 1 else 1
        self.sampler = UniformSampler(self.patch_size)
        
        # variables to be filled later
        self.subjects = None
        self.test_subjects = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.train_queue = None
        self.val_queue = None
        self.test_queue = None
        self.transform = None
        self.preprocess = None

    # Normalization functions
    def get_normalization_transform(self, tr):
        if tr == 'pet_hard_normalization':
            return Lambda(lambda x: x / self.pet_norm)
        elif tr == 'ct_normalization':
            return Lambda(lambda x: (x + 1024.0) / 2000.0)
        else:
            return None

    def prepare_patient_info(self, filename, preprocess_step=None):

        # NUMPY files
        if filename.name.endswith('.npy'):
            rawdata = ScalarImage(filename, reader=numpy_reader)
        # NIFTY, MINC, NRRD, MHA files, or DICOM folder
        else:
            rawdata = ScalarImage(filename)

        if preprocess_step:
            pp = self.get_normalization_transform(preprocess_step)
            return pp(rawdata)
        else:
            return rawdata

    def prepare_patient_data(self, mode='train'):
        """ data is organized as 
            data_dir
                ├── patient1
                |    ├── pet_highdose.nii.gz
                |    ├── pet_lowdose.nii.gz
                |    └── ct.nii.gz
                ├── patient2
                |    ├── pet_highdose.nii.gz
                |    ├── pet_lowdose.nii.gz
                |    └── ct.nii.gz
                ├── ... etc
            """

        # load train/valid patient list in the json file
        patients = load_data_splitting(mode,
                                    self.config['data_split_pkl'],
                                    self.datadir,
                                    fold=self.k,
                                    duplicate_list=self.config['repeat_patient_list'],
                                    quick_test=self.quick_test)

        # create Subject object for each patient
        subjects = []
        for p in patients:
            p_folder = self.datadir.joinpath(p)
            patient_dict = {'id': p}

            for file_type in ['input', 'target']:
                file_info = self.config[file_type + '_files']
                for i in range(len(file_info['name'])):
                    input_path = p_folder.joinpath(file_info['name'][i])
                    transf = file_info['preprocess_step'][i]
                    patient_dict[f"{file_type}{i}"] = self.prepare_patient_info(
                        input_path, transf)

            # Subject instantiation
            s = Subject(patient_dict)
            subjects.append(s)

        return subjects

    def prepare_data(self):

        self.subjects = self.prepare_patient_data('train')
        self.test_subjects = self.prepare_patient_data('test')

    def get_augmentation_transform(self):
        augment = Compose([
            RandomAffine(scales=(0.9, 1.2),                # zoom
                         degrees=5,                         # rotation
                         translation=5,                     # shift
                         isotropic=False,                   # wrt zoom
                         center='image',
                         image_interpolation='linear'),
            RandomFlip(axes=(0, 1, 2))
        ])
        return augment

    def setup(self, stage=None):
        # train/test split subjects
        train_subjects, val_subjects = train_test_split(
            self.subjects, test_size=.2, random_state=42)

        # setup for trainer.fit()
        if stage in (None, 'fit'):
            self.transform = self.get_augmentation_transform() if self.augment else None

            # datasets
            self.train_set = SubjectsDataset(train_subjects, 
                                             transform=self.transform)
            self.val_set = SubjectsDataset(val_subjects)

            # queues
            self.train_queue = Queue(self.train_set,
                                    self.queue_length,
                                    self.samples_per_volume,
                                    self.sampler,
                                    num_workers=self.num_workers)

            self.val_queue = Queue(self.val_set,
                                self.queue_length,
                                self.samples_per_volume,
                                self.sampler,
                                num_workers=self.num_workers)

        # setup for trainer.test()
        if stage in (None, 'test'):
            self.test_set = SubjectsDataset(self.test_subjects)
            self.test_queue = Queue(self.test_set,
                                    self.queue_length,
                                    self.samples_per_volume,
                                    self.sampler,
                                    num_workers=self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_queue, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_queue, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_queue, self.batch_size)


## Generic DataModule without TORCHIO

class GenericDataModule(pl.LightningDataModule):
    def __init__(self, config, quick_test=False):
        super().__init__()
        self.config = config
        self.k = config['k_fold']
        self.quick_test = quick_test
        self.batch_size = self.config['batch_size']
        self.datadir = Path(self.config['data_folder'])
        self.augment = self.config['augment']
        self.num_workers = 4
        # normalization factor for PET data
        self.pet_norm = self.config['pet_normalization_constant']
        
        # variables to be filled later
        self.subjects = None
        self.test_subjects = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.transform = None
        self.preprocess = None

    def prepare_data(self):

        """ data is organized as 
            data_dir
                ├── patient1
                |    ├── pet_highdose.npy
                |    ├── pet_lowdose.npy
                |    └── ct.npy
                ├── patient2
                |    ├── pet_highdose.npy
                |    ├── pet_lowdose.npy
                |    └── ct.npy
                ├── ... etc
            """

        # load train/valid patient list in the json file
        self.subjects = load_data_splitting('train',
                                            self.config['data_split_pkl'],
                                            self.datadir,
                                            fold=self.k,
                                            duplicate_list=self.config['repeat_patient_list'],
                                            quick_test=self.quick_test)
        
        # load train/valid patient list in the json file
        self.test_subjects = load_data_splitting('test',
                                            self.config['data_split_pkl'],
                                            self.datadir,
                                            fold=self.k,
                                            duplicate_list=self.config['repeat_patient_list'],
                                            quick_test=self.quick_test)

    def setup(self, stage=None):
        # train/test split subjects
        train_subjects, val_subjects = train_test_split(
            self.subjects, test_size=.2, random_state=42)

        # setup for trainer.fit()
        if stage in (None, 'fit'):
            # datasets
            self.train_set = DatasetFullVolume(train_subjects, self.config, self.config['augment'])
            self.val_set = DatasetFullVolume(val_subjects, self.config)

        # setup for trainer.test()
        if stage in (None, 'test'):
            self.test_set = DatasetFullVolume(self.test_subjects, self.config)


    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)
    

class DatasetFullVolume(Dataset):
    """ Generates data for Keras """

    def __init__(self, patients, conf=None, augment=False):
        """ Initialization """
        
        self.patients = patients
        self.config = conf
        self.datadir = Path(self.config['data_folder'])
        self.augment = augment

        # normalization factor for PET data
        self.pet_norm = conf['pet_normalization_constant']

        # data shape
        self.full_data_shape = conf['data_shape']
        self.color_channels = conf['color_channels_in']


    def __len__(self):
        'Denotes the total number of samples in the dataset'
        return len(self.patients)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.patients[index]

        # Load data and get label
        X, y = self.data_generation(ID)

        return torch.from_numpy(X).float(), torch.from_numpy(y).float()
    
    def pet_hard_normalization(self, data):
        return data / self.pet_norm

    @staticmethod
    def ct_normalization(data):
        return (data + 1024.0) / 2000.0

    def data_generation(self, patient_id):
        """ Generates data of batch_size samples """
        # X : (batch_size, n_channels, v_size, v_size, v_size)
        
        # this loads and returns the full image size
        X, y = self.load_volume(patient_id)

        return X.astype(np.float64), y.astype(np.float64)

    def load_volume(self, patient_id):

        # initialize input data with correct number of channels
        dat = np.zeros((self.color_channels, *self.full_data_shape))

        # --- Load data and labels
        for i in range(self.color_channels):
            fname = self.datadir.joinpath(patient_id).joinpath(
                self.config['input_files']['name'][i])
            pet = np.memmap(fname, dtype='double', mode='r')
            dat[i, ...] = pet.reshape(self.full_data_shape)
            normalization_func = getattr(self, self.config['input_files']['preprocess_step'][i])
            dat[i, ...] = normalization_func(dat[i, ...])

        # fulldose contains High Dose PET image
        fname2 = self.datadir.joinpath(patient_id).joinpath(
            self.config['target_files']['name'][0])
        target = np.memmap(fname2, dtype='double', mode='r')
        target = target.reshape(1, *self.full_data_shape)
        target = self.pet_hard_normalization(target)

        # manual augmentation of the data half of the time
        if self.augment and np.random.random() < 0.5:
            dat, target = augmenter.random_transform_sample(dat, target)

        return dat, target

