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

def recursive_find_python_class(name, folder=None, current_module="rhtorch.models"):

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