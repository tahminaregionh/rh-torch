#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import random
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import multiprocessing
import socket
import json
import glob
import pytorch_lightning as pl
from torchio import ScalarImage, Subject, SubjectsDataset, Queue
from torchio.transforms import Lambda, RandomAffine, RandomFlip, Compose
from torchio.data import UniformSampler

"""
Generic torchio data generator.
You can optionally extend the functions below.

Assumes data is organized as a folder of input/target files for each patient.

We assume that 'data_split_pkl' in the config file is a pickle file containing
the keys: train_<kfold> and test_<kfold> for each k-fold (e.g. 0..4).
The in-loop validation set is generated from the training set.

"""


class GenericTIODataModule(pl.LightningDataModule):
    def __init__(self, config, quick_test=False):
        super().__init__()
        self.config = config
        self.k = config['k_fold']
        self.quick_test = quick_test
        self.batch_size = self.config['batch_size']
        self.datadir = Path(self.config['data_folder'])
        self.augment = self.config['augment']
        self.num_workers = min(8, multiprocessing.cpu_count()) \
            if not socket.gethostname().startswith('ibm') \
            else 16  # Could do 32 (has 132) but might run out..

        # for the queue
        self.patch_size = self.config['patch_size']
        if max(self.patch_size) > 96:
            self.samples_per_volume = 12  # Emperically selected.
        else:
            self.samples_per_volume = 64
        self.queue_length = self.samples_per_volume*self.batch_size
        self.sampler = UniformSampler(self.patch_size)

        # variables to be filled later
        self.subjects = None
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.train_set = None
        self.val_set = None
        self.train_queue = None
        self.val_queue = None
        self.transform = None
        self.preprocess = None

    def numpy_reader(self, path):
        return np.load(path), np.eye(4)

    def load_data_splitting(self, mode="train"):

        # search for pkl/json file in data folder if none passed
        if pkl_file := self.config['data_split_pkl']:
            pkl_file = self.datadir.joinpath(pkl_file)
        else:
            pkl_file = \
                glob.glob(f"{self.datadir}/*_train_test_split_*_fold.json")[0]
        if not pkl_file.exists():
            raise FileNotFoundError(
                "Data train/test split info file not found. "
                "Add file to data folder or declare in config file.")

        # use json or pickle loader depending on file extension
        load_module = json if pkl_file.name.endswith(".json") else pickle
        load_settings = 'r' if pkl_file.name.endswith(".json") else 'rb'
        with open(pkl_file, load_settings) as f:
            split_data_info = load_module.load(f)

        patients = split_data_info[f"{mode}_{self.k}"]
        if self.config['repeat_patient_list']:
            patients = np.repeat(patients, self.config['repeat_patient_list'])

        # trim the test/train set if quick test
        if self.quick_test:
            keep = 2 if mode == 'test' else 10
            patients = patients[:keep]

        return patients

    # Normalization functions
    def get_normalization_transform(self, tr):
        """ Each normalization must have an inv_normalization applied
            in de_normalize """
        if tr == 'ct_normalization':  # Left as example. Overwrite if needed
            return Lambda(lambda x: (x + 1024.0) / 2000.0)
        elif tr == 'inv_ct_normalization':
            return Lambda(lambda x: x*2000.0 - 1024.0)
        else:
            return None

    # Helper function to invert the normalization performed during loading
    def de_normalize(self, img, transform):
        trans = self.get_normalization_transform(f"inv_{transform}")
        return img if trans is None else trans(img)

    def prepare_patient_info(self, filename, preprocess_step=None):

        # NUMPY files
        if filename.name.endswith('.npy'):
            rawdata = ScalarImage(filename, reader=self.numpy_reader)
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
        # load train/valid patient list in the json/pkl file
        patients = self.load_data_splitting(mode)

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
                    patient_dict[f"{file_type}{i}"] = \
                        self.prepare_patient_info(input_path, transf)

            # Subject instantiation
            s = Subject(patient_dict)
            subjects.append(s)

        return subjects

    def prepare_data(self):

        self.subjects = self.prepare_patient_data('train')
        # Set up test_subject only for inference.
        self.test_subjects = self.prepare_patient_data('test')

        # train/test split subjects
        self.train_subjects, self.val_subjects = train_test_split(
            self.subjects, test_size=.1, random_state=42)
        assert len(self.val_subjects) > 0

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

        # setup for trainer.fit()
        if stage in (None, 'fit'):
            self.transform = self.get_augmentation_transform() \
                if self.augment else None

            # datasets
            self.train_set = SubjectsDataset(self.train_subjects,
                                             transform=self.transform)
            self.val_set = SubjectsDataset(self.val_subjects)

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
        if stage == 'test':
            self.test_set = SubjectsDataset(self.test_subjects)

    def train_dataloader(self):
        return DataLoader(self.train_queue, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_queue, self.batch_size)


"""
Generic data generator.
You can optionally extend the functions below.

Assumes data is organized as a folder of patches extracted for each patient.
The reference patch is stored under: datafolder/gt/ProjectID_PatientID_PatchID.npy
where ProjectID_PatientID is e.g. "FET_001", and PatchID is a counter of the extracted patches.
The corresponding input patches are stored under the same in datafolder/patches.

At runtime, 1 random patch is chosen for the patient in the minibatch.
To repeat the list of patients, use 'repeat_patient_list' in the config file.

We assume that 'data_split_pkl' in the config file is a pickle file containing the keys: train_<kfold> and test_<kfold>
for each k-fold (e.g. 0..4). The in-loop validation set is generated from the training set.

"""


class DatasetPreprocessedRandomPatches(Dataset):
    """ Generates data for Keras """

    def __init__(self, data_type='train', conf=None, augment=False, test=False):
        """ Initialization """

        self.rootdir = conf['data_folder']
        self.config = conf
        self.data_type = data_type
        self.augment = augment

        # load patient list in the json file
        self.patients = []
        self.patches = {}
        self.repeat_list = 1 if not 'repeat_patient_list' in list(
            conf.keys()) else conf['repeat_patient_list']
        self.load_splits(test)

        # set color channels
        self.color_channels = self.config['color_channels_in']

        # keep track of the original data shape
        # add the color channel (channel_first in Pytorch)
        self.data_shape_in = [self.color_channels, *self.full_data_shape]
        self.data_shape_out = [1, *self.full_data_shape]

    # loading and reading data splitting
    def load_splits(self, test_set=False):
        if self.config['data_split_pkl'].endswith('.pkl') or self.config['data_split_pkl'].endswith('.pickle'):
            with open(self.config['data_split_pkl'], 'rb') as f:
                split_data_info = pickle.load(f)
        elif self.config['data_split_pkl'].endswith('.yaml'):
            pass  # to be implemented

        if self.data_type == 'test':
            self.patients = split_data_info[f"test_{self.config['k_fold']}"]
        else:
            train, valid = train_test_split(
                split_data_info[f"train_{self.config['k_fold']}"], test_size=.2, random_state=42)
            self.patients = train if self.data_type == 'train' else valid

            # Load list of patches
            for file in os.listdir(self.rootdir+'/gt'):
                ID = '_'.join(file.split('_')[:2])
                if ID in self.patients:
                    if not ID in self.patches:
                        self.patches[ID] = []
                    self.patches[ID].append(file)
            # Remove patients without any patches
            self.patients = [p for p in self.patients if p in self.patches]

            # REPEAT
            self.patients = np.repeat(self.patients, self.repeat_list)

        self.full_data_shape = self.config["data_shape"]

        # trim the test/train set if test
        if test_set:
            keep = 10
            if self.data_type == 'valid':
                keep = 2
            self.patients = self.patients[:keep]

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

    def data_generation(self, patient_id):
        """ Generates data of batch_size samples """
        # X : (batch_size, n_channels, v_size, v_size, v_size)

        X, y = self.load_patch(patient_id)

        return X.astype(np.float64), y.astype(np.float64)

    def load_patch(self, PID):

        # Get patch
        patch = random.choice(self.patches[PID])

        X = np.load(os.path.join(self.rootdir, 'patches', patch))
        y = np.load(os.path.join(self.rootdir, 'gt', patch))

        # Patches should be channels first. If not, use the below to swap
        #X = np.rollaxis(X,-1)
        #y = np.rollaxis(y,-1)

        # Simple augmentation. OBS: fliplr works the same for pytorch as TF?
        if self.augment:
            # Augment data
            r = np.random.random()
            if r < 0.2:
                X = np.rot90(X, 1, axes=(2, 3))  # rot 90
                y = np.rot90(y, 1, axes=(1, 3))
            elif r < 0.4:
                X = np.rot90(X, 3, axes=(2, 3))  # rot -90
                y = np.rot90(y, 3, axes=(2, 3))
            elif r < 0.6:
                X = np.rot90(X, 2, axes=(2, 3))  # flip UD
                y = np.rot90(y, 2, axes=(2, 3))
            elif r < 0.8:
                X = np.fliplr(X)  # flip LR
                y = np.fliplr(y)

        # Normalize
        X, y = self.normalize(X, y)

        return X, y

    # Function to load the full volume of a patient. Has to be overloaded
    def load_full_volume(self, patient_id):
        sys.exit(
            'This function has to be overloaded when extending the generic function')

    def normalize(self, in_img, out_img):
        # Do nothing
        return in_img, out_img

    def de_normalize(self, image):
        # Do nothing
        return image
