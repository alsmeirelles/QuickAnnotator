# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import sys
import os
import time
import re
import multiprocessing
import argparse

from make_tile_for_patch import pf_form

#Segframe import
sys.path.append('../Segframe')
from Preprocessing import PImage

def _run_dir(path):

    rex = re.compile(pf_form)
    patches = list(filter(lambda f:f.endswith('.png'),os.listdir(path)))

    rt = []
    for p in patches:
        pm = rex.match(p)
        if pm is None:
            continue
        seg = PImage(os.path.join(path,p),keepImg=False,origin=pm.group('tcga'),coord=(pm.group('x'),pm.group('y')),
                         verbose=1)
        rt.append(seg)

    return rt

def generate_set(path,n,processes=2):
    dlist = []
    files = os.listdir(path)

    dirs = list(filter(lambda i:os.path.isdir(os.path.join(path,i)),files))
    multi_dir = True if len(dirs) > 0 else False

    rt = []
    if multi_dir:
        results = None
        with multiprocessing.Pool(processes=processes) as pool:
            results = [pool.apply_async(_run_dir,(os.path.join(path,d),)) for d in dirs]
            for r in results:
                rt.extend(r.get())
    else:
        rt = _run_dir(path)

    return rt
    
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract tiles from a WSI \
        discarding background.')

    parser.add_argument('-mp', dest='mp', type=int, 
        help='Use multiprocessing. Number of processes to spawn', default=2,required=False)
    parser.add_argument('-set', dest='set', type=int, 
        help='Number of patches to select for initial training set', default=500,required=False)    
    parser.add_argument('-pd', dest='patches_dir', type=str, default=None, 
        help='Patch location.')
    
    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.patches_dir):
        sys.exit()

    rt = generate_set(config.patches_dir,config.set,config.mp)
        
    print("Total patches available: {}".format(len(rt)))

