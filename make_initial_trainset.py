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
import pickle

from make_tile_for_patch import pf_form

#Segframe import
#sys.path.append('../Segframe')
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

def makeImg(path,keep=False):
    """
    Returns a PImage object for a given file
    """

    file_name = os.path.basename(path)
    rex = re.compile(pf_form)
    pm = rex.match(file_name)
    img = None
    if not pm is None:
        img = PImage(path,keepImg=keep,origin=pm.group('tcga'),coord=(pm.group('x'),pm.group('y')),
                         verbose=1)

    return img

def generate_set(path,n,nval,cache='.cache',processes=2):
    """
    Returns and ndarray of randomly selected items from data pool
    """
    dlist = []
    files = os.listdir(path)

    dirs = list(filter(lambda i:os.path.isdir(os.path.join(path,i)),files))
    multi_dir = True if len(dirs) > 0 else False

    if not os.path.isdir(cache):
        os.mkdir(cache)

    rt = []
    if multi_dir:
        results = None
        with multiprocessing.Pool(processes=processes) as pool:
            results = [pool.apply_async(_run_dir,(os.path.join(path,d),)) for d in dirs]
            for r in results:
                rt.extend(r.get())
    else:
        rt = _run_dir(path)

    #Save pool metadata to cache
    rt = np.asarray(rt)
    train_idx = np.random.choice(len(rt),n,replace=False)
    train_set = rt[train_idx]
    with open(os.path.join(cache,'initial_train.pik'),'wb') as fd:
        pickle.dump(train_set,fd)

    #Select validation patches if defined
    val_idx = None
    val_set = None
    if nval > 0:
        val_idx = val_idx = np.random.choice(np.setdiff1d(np.arange(rt.shape[0]),train_idx),nval,replace=False)
        val_set = rt[val_idx]
        with open(os.path.join(cache,'val_set.pik'),'wb') as fd:
            pickle.dump(val_set,fd)
    #Update pool
    remove = train_idx if val_idx is None else np.concatenate((train_idx,val_idx),axis=0) 
    pool = np.delete(rt,remove)
    with open(os.path.join(cache,'pool.pik'),'wb') as fd:
        pickle.dump(pool,fd)
    
    return (train_set,val_set)
    
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract tiles from a WSI \
        discarding background.')

    parser.add_argument('-mp', dest='mp', type=int, 
        help='Use multiprocessing. Number of processes to spawn', default=2,required=False)
    parser.add_argument('-set', dest='set', type=int, 
        help='Number of patches to select for initial training set', default=500,required=False)
    parser.add_argument('-valset', dest='valset', type=int, 
        help='Number of patches to select for validation set', default=100,required=False)        
    parser.add_argument('-pd', dest='pool', type=str, default=None, 
        help='Patch location.')
    
    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.pool):
        sys.exit()

    rt = generate_set(config.pool,config.set,config.valset,cache='cache',processes=config.mp)
    
    print("Training patches: {}".format(len(rt[0])))

