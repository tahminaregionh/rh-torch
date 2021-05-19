#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import random, os, sys
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

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
        self.repeat_list = 1 if not 'repeat_patient_list' in list(conf.keys()) else conf['repeat_patient_list']
        self.load_splits(test)
        
        # set color channels
        self.color_channels = self.config['color_channels_in']
            
        # keep track of the original data shape
        self.data_shape_in = [self.color_channels, *self.full_data_shape]     # add the color channel (channel_first in Pytorch)
        self.data_shape_out = [1, *self.full_data_shape]
            
    # loading and reading data splitting
    def load_splits(self, test_set=False):
        if self.config['data_split_pkl'].endswith('.pkl') or self.config['data_split_pkl'].endswith('.pickle'):
            with open(self.config['data_split_pkl'], 'rb') as f:
                split_data_info = pickle.load(f)
        elif self.config['data_split_pkl'].endswith('.yaml'):
            pass # to be implemented
            
        if self.data_type == 'test':
            self.patients = split_data_info[f"test_{self.config['k_fold']}"]
        else:
            train,valid = train_test_split(split_data_info[f"train_{self.config['k_fold']}"],test_size=.2,random_state=42)
            self.patients = train if self.data_type == 'train' else valid
            
            # Load list of patches
            for file in os.listdir(self.rootdir+'/gt'):
                ID = '_'.join(file.split('_')[:2])
                if ID in self.patients:
                    if not ID in self.patches:
                        self.patches[ID] = []
                    self.patches[ID].append( file )
            # Remove patients without any patches
            self.patients = [ p for p in self.patients if p in self.patches ]
            
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
        
        X, y = self.load_patch(patient_id)     

        return X.astype(np.float64), y.astype(np.float64)
    
    def load_patch(self, PID):
        
        # Get patch
        patch = random.choice( self.patches[PID] )
        
        X = np.load(os.path.join(self.rootdir,'patches',patch))
        y = np.load(os.path.join(self.rootdir,'gt',patch))
        
        # Patches should be channels first. If not, use the below to swap
        #X = np.rollaxis(X,-1)
        #y = np.rollaxis(y,-1)
        
        # Simple augmentation. OBS: fliplr works the same for pytorch as TF?
        if self.augment:
            # Augment data
            r = np.random.random()
            if r < 0.2:
                X = np.rot90(X,1,axes=(2,3)) # rot 90
                y = np.rot90(y,1,axes=(1,3))
            elif r < 0.4:
                X = np.rot90(X,3,axes=(2,3)) # rot -90
                y = np.rot90(y,3,axes=(2,3))
            elif r < 0.6:
                X = np.rot90(X,2,axes=(2,3)) # flip UD
                y = np.rot90(y,2,axes=(2,3))
            elif r < 0.8:
                X = np.fliplr(X) # flip LR
                y = np.fliplr(y)
                
        # Normalize
        X,y = self.normalize(X,y)
        
        return X,y
    
    # Function to load the full volume of a patient. Has to be overloaded
    def load_full_volume(self, patient_id):
        sys.exit('This function has to be overloaded when extending the generic function')
    
    def normalize ( self, in_img, out_img ):
        # Do nothing
        return in_img, out_img
    
    def de_normalize( self, image ):
        # Do nothing
        return image
    