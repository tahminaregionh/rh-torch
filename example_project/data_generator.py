#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import random, os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

""" 
Example low-dose PET to PET project with preprocessed patches of data.

Assumes data is organized as a folder of patches extracted for each patient.
The reference patch is stored under: datafolder/gt/ProjectID_PatientID_PatchID.npy
where ProjectID_PatientID is e.g. "FET_001", and PatchID is a counter of the extracted patches.
The corresponding input patches are stored under the same in datafolder/patches.

At runtime, 1 random patch is chosen for the patient in the minibatch. 
To repeat the list of patients, use 'repeat_patient_list' in the config file.

We assume that 'data_split_pkl' in the config file is a pickle file containing the keys: train_<kfold> and test_<kfold> 
for each k-fold (e.g. 0..4). The in-loop validation set is generated from the training set.

"""
    
class DatasetPreprocessedPatches(Dataset):
    """ Generates data for Keras """
    def __init__(self, data_type='train', conf=None, augment=False, test=False):
        """ Initialization """
        
        self.rootdir = conf['data_folder']
        self.config = conf
        self.data_type = data_type
        self.augment = augment
        # normalization factor for PET data
        self.pet_norm = conf['pet_normalization_constant'] 
        
        # load patient list in the json file 
        self.patients = []
        self.patches = {}
        self.repeat_list = 1 if not 'repeat_patient_list' in list(conf.keys()) else conf['repeat_patient_list']
        self.load_pkl(test)
        
        # set color channels
        self.color_channels = self.config['color_channels_in']
            
        # keep track of the original data shape
        self.data_shape_in = [self.color_channels, *self.full_data_shape]     # add the color channel (channel_first in Pytorch)
        self.data_shape_out = [1, *self.full_data_shape]
            
    # loading and reading data splitting
    def load_pkl(self, test_set=False):
        with open(self.config['data_split_pkl'], 'rb') as f:
            split_data_info = pickle.load(f)
            
        if self.data_type == 'test':
            self.patients = split_data_info[f"test_{self.config['k_fold']}"]
        else:
            train,valid = train_test_split(split_data_info[f"train_{self.config['k_fold']}"],test_size=.2,random_state=42)
            self.patients = train if self.data_type == 'train' else valid
            
            # Load patches
            for PID in self.patients:
                self.patches[PID]=[p.name for p in Path(self.rootdir+'/gt').glob(f'{PID}_*.npy')]
            
            # REPEAT
            self.patients = np.repeat(self.patients,self.repeat_list)
            
            
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
        
        LD, HD = self.load_patch(patient_id)     

        return LD.astype(np.float64), HD.astype(np.float64)
    
    def load_patch(self, PID):
        
        # Get patch
        patch = random.choice( self.patches[PID] )
        
        X = np.load(os.path.join(self.rootdir,'patches',patch))
        y = np.load(os.path.join(self.rootdir,'gt',patch))
        
        # Simple augmentation. OBS, did not check if this works for pytorch..
        if self.augment:
            # Augment data
            r = np.random.random()
            if r < 0.2:
                X = np.rot90(X,1,axes=(1,2)) # rot 90
                y = np.rot90(y,1,axes=(1,2))
            elif r < 0.4:
                X = np.rot90(X,3,axes=(1,2)) # rot -90
                y = np.rot90(y,3,axes=(1,2))
            elif r < 0.6:
                X = np.flipud(X) # flip UD
                y = np.flipud(y)
            elif r < 0.8:
                X = np.fliplr(X) # flip LR
                y = np.fliplr(y)
                
        # Normalize
        X = (X - 0.0) / (self.pet_norm - 0.0) 
        y = (y - 0.0) / (self.pet_norm - 0.0) 
        
        X = X.reshape(self.data_shape_in)
        y = y.reshape(self.data_shape_out)
        
        return X,y
    
    """ Example function to load the full volume.
    In this case, only a subset of the image is used, and the dose (25%) is used to scale the image.
    This is project, specific!
    """
    def load_full_volume(self, patient_id):
        import pyminc.volumes.factory as pyminc
        print("Loading volume from DatasetPreprocessedPatches")
        mnc = pyminc.volumeFromFile(f'{self.rootdir}/../mnc/{patient_id}/PET_25.mnc')
        img = np.array(mnc.data,dtype='double')
        mnc.closeVolume()
        
        img = img[:,72:328,72:328]
        img = img*(100/25) # Correct for dose reduction
        
        # Normalize
        img = (img - 0.0) / (self.pet_norm - 0.0) 

        return img
    
    def de_normalize(self, image ):
        return image * self.pet_norm
    
