# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:12:27 2021

@author: Claes Ladefoged

Example DataGenerator that extends the generic data generator.
Overwrite only the functions that differs.

Important: You must name this function "data_generator.py" and place it in the project root folder.

"""

from rhtorch.data.data_generator import DatasetPreprocessedRandomPatches

class CustomDataLoader(DatasetPreprocessedRandomPatches):
    def __init__(self, data_type='train', conf=None, augment=False, test=False):
        super().__init__(data_type,conf,augment,test)
        self.pet_normalization = conf['pet_normalization']
   
    def normalize( self, in_img, out_img ):  
        out_img = out_img / self.pet_normalization
        return in_img, out_img
    
    def de_normalize( self, image ):  
        image = image * self.pet_normalization
        return image