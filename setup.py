#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='rhtorch',
     version='0.1.3',
     author="Claes Ladefoged",
     author_email="claes.noehr.ladefoged@regionh.dk",
     description="Scripts used at CAAI for torch training",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/CAAI/rh-torch",
     entry_points={
          'console_scripts': [
              'torch_training = rhtorch.torch_training:main',
          ]
     },
     packages=setuptools.find_packages(),
     install_requires=[
         'torch',
         'pytorch_lightning',
         'torchio',
         'scikit-learn',
         'wandb',
         'ruamel.yaml',
         'PyWavelets',
         'monai'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
 )
