#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import argparse
from pathlib import Path
import numpy as np
from numpy.lib import stride_tricks
from caai.pytorchlightning.models import modules
from caai.pytorchlightning.config_utils import load_model_config
import data_generator

def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6

def predict_patches( X, w):
    wi,wj,wk=w
    si = wi//4 # 2
    sj = wj//4 # 16
    sk = wk//4 # 16
    
    predicted = np.zeros(X.shape)
    predicted_counter = np.zeros(X.shape)

    patches = cutup(X,(wi,wj,wk),(si,sj,sk))
    
    for i in range(patches.shape[0]):
        print(i,'/',patches.shape[0])
        fi = i*si
        for j in range(patches.shape[1]):
            fj = j*sj
            for k in range(patches.shape[2]):
                fk = k*sk
                patch = np.reshape( patches[i,j,k,...], (1,wi,wj,wk))
                torch_patch = torch.from_numpy( patch ).float().unsqueeze(0)
                if args.GPU:
                    torch_patch = torch_patch.cuda()
                y_hat = model( torch_patch )
                if args.GPU:
                    y_hat = y_hat.cpu()
                reduced_fov_patch = np.reshape(y_hat.detach().numpy()[0],(wi,wj,wk))[si:wi-si,sj:wj-sj,sk:wk-sk]
                predicted[fi+si:fi+wi-si,fj+sj:fj+wj-sj,fk+sk:fk+wk-sk] += reduced_fov_patch
                predicted_counter[fi+si:fi+wi-si,fj+sj:fj+wj-sj,fk+sk:fk+wk-sk] += 1.0
    return predicted, predicted_counter

def predict_slices( X, wi ):
    si = wi//4
    wj = wk = X.shape[1]
    sj = sk = 1
    
    predicted = np.zeros(X.shape)
    predicted_counter = np.zeros(X.shape)

    patches = cutup(X,(wi,wj,wk),(si,sj,sk))
    
    for i in range(patches.shape[0]):
        print(i,'/',patches.shape[0])
        fi = i*si
        
        patch = np.reshape( patches[i,0,0,...], (1,wi,wj,wk))
        torch_patch = torch.from_numpy( patch ).float().unsqueeze(0)
        if args.GPU:
            torch_patch = torch_patch.cuda()
        y_hat = model( torch_patch )
        if args.GPU:
            y_hat = y_hat.cpu()
        predicted[fi:fi+wi,:,:] += np.reshape(y_hat.detach().numpy()[0],(wi,wj,wk))
        predicted_counter[fi:fi+wi,:,:] += 1.0
    return predicted, predicted_counter

def predict( pt ):
    
    X = valid_test_gen.load_full_volume(pt)
    
    if configs['data_shape'][1] < X.shape[1] or configs['data_shape'][2] < X.shape[2]:
        # Using patches rather than full slices
        predicted, predicted_counter = predict_patches( X, configs['data_shape'] )
    elif configs['data_shape'][1] == X.shape[1] and configs['data_shape'][2] and X.shape[2]:
        # YZ is same as X second and third dim. Assume slice-based
        predicted, predicted_counter = predict_slices( X, configs['data_shape'][0] )
        

    predicted_counter = np.maximum( predicted_counter, 1.0 ) # Do not divide by 0..
    predicted = np.true_divide( predicted, predicted_counter )
    
    return predicted

def save_mnc( patient, img ):
    import pyminc.volumes.factory as pyminc
    
    # save dir
    p = Path(f'Data/mnc/{patient}/preds')
    p.mkdir(parents=True,exist_ok=True)
    
    # Denormalize
    denormalized_img = valid_test_gen.de_normalize( y_pred ) 
    
    out = pyminc.volumeLikeFile(str(p.parent.joinpath('PET_100.mnc')),str(p.joinpath(f'{model_outname}.mnc')))
    out.data = denormalized_img
    out.writeFile()
    out.closeVolume()
    print("Saved file to:",str(p.joinpath(f'{model_outname}.mnc')))
    
def get_model_outname():
    outname = f"{model_outname}_k{args.kfold}_e{configs['epoch']}"
    if not project_dir.joinpath('trained_models',outname).is_dir():
        # check for multiGPU
        bz_1gpu = 'bz{:d}'.format(configs['batch_size'])
        for numGPUs in range(2,5):
            bz_multiGPU = 'bz{:d}'.format(configs['batch_size']*numGPUs)
            outname_mGPU = outname.replace(bz_1gpu,bz_multiGPU)
            if project_dir.joinpath('trained_models',outname_mGPU).is_dir():
                return outname_mGPU
        print("No such model",outname)
        exit(-1)
        
    else:
        return outname
    
parser = argparse.ArgumentParser(description='Runs training on dataset in the input directory and config.yaml')
parser.add_argument("-i", "--input", 
                    help="Directory containing data json. Same folder contains 'config.yaml' file. Will use current working directory if nothing passed", 
                    type=str, default=os.getcwd())
parser.add_argument("-l", "--log", help="Key used to select data (Options: valid, test)", type=str, default='valid')
parser.add_argument("-k", "--kfold", help="K-fold", type=int, default=0)
parser.add_argument("-c", "--config", help="Config file else than {project_name}_config.yaml in input folder", type=str, default='')
parser.add_argument("--GPU",action="store_true", default=False)
args = parser.parse_args()

project_dir = Path(args.input)
configs, model_outname = load_model_config(project_dir, args.config)
configs['k_fold'] = args.kfold
model_outname = get_model_outname()
configs['project_dir'] = str(project_dir)

# Data
data_generator = getattr(data_generator, configs['data_generator'])
valid_test_gen = data_generator(args.log, conf=configs)

# Load model and checkpoint
checkpoint_path = project_dir.joinpath('trained_models',model_outname,'checkpoints','Checkpoint_min_val_loss-v2.ckpt')
model = getattr(modules, configs['module'])
model = model.load_from_checkpoint(checkpoint_path,hparams=configs,in_shape=valid_test_gen.data_shape_in)
model.eval()

if args.GPU:
    model.cuda()
    
for c,patient in enumerate(valid_test_gen.patients):
    if not os.path.exists(f'Data/mnc/{patient}/preds/{model_outname}.mnc'):
            print(patient)
            predicted = predict( patient )
            save_mnc( patient, predicted )
        
    