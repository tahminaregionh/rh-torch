# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:12:27 2021
@author: Claes Ladefoged
"""
from torchio import SubjectsDataset, Queue
from torchio.transforms import (
    Lambda,
    Compose,
    CropOrPad,
    Resample,
    ToCanonical
)
from sklearn.model_selection import train_test_split
from rhtorch.data.data_generator import (
    GenericTIODataModule,
    DatasetPreprocessedRandomPatches
)

"""
Example TIODataModule that extends GenericTIODataModule from rhtorch.
Overwrite only the functions that differs.

Important: You must name this file "data_generator.py"
           and place it in the project root folder.

Example changes added below:
 ## __init__ ##
    * Overwritten samples_per_volume and queue_length
    * Defined a constant for PET normalization
 ## get_normalization_transform ##
    * Defined the function for PET normalization used during data preprocessing
 ## setup ##
    * Overwritten the function, extending with added spatial transformation
      applied to all files. The entire setup function was copied before adding
      the differences.
"""


class ExampleTIODataModule(GenericTIODataModule):
    def __init__(self, config, quick_test=False):
        super().__init__(config, quick_test)

        self.samples_per_volume = 64
        self.queue_length = self.samples_per_volume*self.batch_size

        # normalization factor for PET data
        self.pet_norm = self.config['pet_normalization_constant']

    # Normalization functions
    def get_normalization_transform(self, tr):
        # Hint: If you only load pre-normalized files, you can still invert
        # normalization here, by only declaring an inv_<...> clause.
        if tr == 'ct_normalization':
            return Lambda(lambda x: (x + 1024.0) / 2000.0)
        elif tr == 'inv_ct_normalization':
            return Lambda(lambda x: x*2000.0 - 1024.0)
        elif tr == 'pet_hard_normalization':
            return Lambda(lambda x: x / self.pet_norm)
        elif tr == 'inv_pet_hard_normalization':
            return Lambda(lambda x: x * self.pet_norm)
        else:
            return None

    # Redefined setup to add preprocessing_transform on all data.
    def setup(self, stage=None):
        # train/test split subjects
        train_subjects, val_subjects = train_test_split(
            self.subjects, test_size=.2, random_state=42)

        # Add spatial preprocessing
        self.preprocessing_transform = Compose([ToCanonical(),
                                                Resample(2),
                                                CropOrPad((192, 192, 400))])

        # setup for trainer.fit()
        if stage in (None, 'fit'):
            self.augmentation_transform = self.get_augmentation_transform() \
                                          if self.augment else None

            self.transform = Compose([self.preprocessing_transform,
                                      self.augmentation_transform])

            # datasets
            self.train_set = SubjectsDataset(
                train_subjects, transform=self.transform)
            self.val_set = SubjectsDataset(
                val_subjects, transform=self.preprocessing_transform)

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
            self.test_set = SubjectsDataset(
                self.test_subjects, transform=self.preprocessing_transform)


"""
Example DataGenerator (without torchio) that extends the generic data generator
Overwrite only the functions that differs.

Important: You must name this function "data_generator.py"
           and place it in the project root folder.
"""


class CustomDataLoader(DatasetPreprocessedRandomPatches):
    def __init__(self, data_type='train', conf=None,
                 augment=False, test=False):
        super().__init__(data_type, conf, augment, test)
        self.pet_normalization_constant = conf['pet_normalization_constant']

    def normalize(self, in_img, out_img):
        out_img = out_img / self.pet_normalization_constant
        return in_img, out_img

    def de_normalize(self, image):
        image = image * self.pet_normalization_constant
        return image
