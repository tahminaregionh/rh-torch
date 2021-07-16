# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:40:15 2021

@author: clad0003
"""

import rhtorch
import pkgutil
import os
import sys
import importlib
from pathlib import Path
from typing import Union


def recursive_find_python_class(name, folder=None,
                                current_module="rhtorch.models"):

    # Set default search path to root modules
    if folder is None:
        folder = [os.path.join(rhtorch.__path__[0], 'models')]

    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + '.' + modname)
            if hasattr(m, name):
                tr = getattr(m, name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + '.' + modname
                tr = recursive_find_python_class(name, folder=[os.path.join(
                    folder[0], modname)], current_module=next_current_module)

            if tr is not None:
                break

    if tr is None:
        sys.exit(f"Could not find module {name}")

    return tr


def find_best_checkpoint(ckpt_dir: Union[str, Path],
                         num_saved_checkpoints: int = 3):
    """
    Args:
        ckpt_dir
            Directiory to the .ckpt files stored during training
        num_saved_checkpoints:
            Number of stored checkpoint, specified in the callback
    """
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    import torch
    import numpy as np
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)
    # Get all checkpoints except last.ckpt
    paths = [p for p in ckpt_dir.iterdir() if not p.name == 'last.ckpt']
    if len(paths) == num_saved_checkpoints:
        # Read the best from the last.ckpt
        return torch.load(ckpt_dir / 'last.ckpt')['callbacks'][ModelCheckpoint]['best_model_path']
    else:
        # Training was performed over several runs, resulting in multiple
        # "best" checkpoints saved. Need to run through them all to see which
        # one is was in fact the best
        best_score = np.inf
        best_path = None
        for p in paths:
            ckpt = torch.load(p)['callbacks'][ModelCheckpoint]
            if (val_loss := ckpt['best_model_score'].tolist()) < best_score:
                best_score = val_loss
                best_path = ckpt['best_model_path']
        return best_path
