#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
     name='rhtorch',  
     version='0.0.1',
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
         'wandb',
         'ruamel.yaml'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
 )