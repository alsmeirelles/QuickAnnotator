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
import os,sys
import importlib
import math

from types import SimpleNamespace
from keras.preprocessing.image import ImageDataGenerator

#AL System imports
from AL.Common import load_model_weights
from Trainers import ThreadedGenerator
from Trainers.GenericTrainer import Trainer
from Utils.CacheManager import CacheManager


def run_active_learning(pool,data,qa_config,proj_path,iteration):

    kwargs = {}
    kwargs['tnet'] = qa_config.get("active_learning","tnet")
    kwargs['net'] = qa_config.get("active_learning","net")
    kwargs['data'] = "CellRep"
    kwargs['predst'] = qa_config.get("common","pool")
    kwargs['batch_size'] = qa_config.getint("train_ae","batchsize")
    kwargs['learn_r'] = 0.0005
    kwargs['epochs'] = qa_config.getint("active_learning","alepochs")
    kwargs['split'] = (0.9,0.01,0.09)
    kwargs['train_set'] = qa_config.getint("active_learning","initial_set")
    kwargs['val_set'] = qa_config.getint("active_learning","val_size")
    kwargs['phi'] = qa_config.getint("active_learning","phi")
    kwargs['tnphi'] = qa_config.getint("active_learning","tnphi")
    kwargs['tdim'] = (qa_config.getint("active_learning","input_size"),)*2
    kwargs['strategy'] = qa_config.get("active_learning","strategy")
    kwargs['weights_path'] = os.path.join(proj_path,'models')
    kwargs['new_net'] = True
    kwargs['gpu_count'] = qa_config.getint("cuda","gpucount")
    kwargs['cpu_count'] = qa_config.getint("pooling","npoolthread")
    kwargs['bdir'] = os.path.join(proj_path,'cache')
    kwargs['cache'] = os.path.join(proj_path,'cache')
    kwargs['verbose'] = config.getint("common","verbose")
    kwargs['logdir'] = os.path.join(proj_path,'cache')
    kwargs['progressbar'] = False
    kwargs['keepimg'] = True
    kwargs['delay_load'] = True
    kwargs['debug'] = False
    kwargs['model_path'] = os.path.join(proj_path,'models')
    kwargs['save_dt'] = False
    kwargs['spool'] = 2
    kwargs['pred_size'] = 15000
    kwargs['save_w'] = False
    kwargs['acquire'] = qa_config.getint("active_learning","aq_size")
    kwargs['clusters'] = 20
    kwargs['ffeat'] = None
    kwargs['recluster'] = 0
    kwargs['ac_function'] = 'dada'
    kwargs['un_function'] = qa_config.get("active_learning","un_function")

    config = SimpleNamespace(**kwargs)
    
    files = {
        'datatree.pik':os.path.join(config.cache,'{}-datatree.pik'.format(config.data)),
        'tcga.pik':os.path.join(config.cache,'tcga.pik'),
        'metadata.pik':os.path.join(config.cache,'{0}-{1}-metadata.pik'.format(config.data,os.path.basename(config.predst))),
        'un_metadata.pik':os.path.join(config.cache,'{0}-{1}-un_metadata.pik'.format(config.data,os.path.basename(config.predst))),
        'sampled_metadata.pik':os.path.join(config.cache,'{0}-sampled_metadata.pik'.format(config.data)),
        'testset.pik':os.path.join(config.cache,'{0}-testset.pik'.format(config.data)),
        'initial_train.pik':os.path.join(config.cache,'{0}-inittrain.pik'.format(config.data)),
        'split_ratio.pik':os.path.join(config.cache,'{0}-split_ratio.pik'.format(config.data)),
        'clusters.pik':os.path.join(config.cache,'{0}-clusters.pik'.format(config.data)),
        'data_dims.pik':os.path.join(config.cache,'{0}-data_dims.pik'.format(config.data)),
        'tiles.pik':os.path.join(config.predst,'tiles.pik'),
        'test_pred.pik':os.path.join(config.logdir,'test_pred.pik'),
        'cae_model.h5':os.path.join(config.model_path,'cae_model.h5'),
        'vgg16_weights_notop.h5':os.path.join('PretrainedModels','vgg16_weights_notop.h5')}

    cache_m = CacheManager(locations=files)

    dsm = importlib.import_module('Datasources',config.data)
    ds = getattr(dsm,config.data)(config.predst,config.keepimg,config)
    ts = importlib.import_module('Trainers',config.strategy)
    trainer = getattr(ts,config.strategy)(config)

    trainer.train_x = data[0][:kwargs['val_set']]
    trainer.train_y = data[1][:kwargs['val_set']]
    trainer.val_x = data[0][kwargs['val_set']:]
    trainer.val_y = data[1][kwargs['val_set']:]

    params = {}
    
    model = trainer.load_modules(config.net,ds)
    model.setPhi(config.phi)
    params['model'] = model
    params['acquisition'] = iteration    
    tmodel,sw_thread,_ = trainer._target_net_train(model)
    params['sw_thread'] = sw_thread[0] if len(sw_thread) == 1 else sw_thread
    params['config'] = config
    params['emodels'] = tmodel
    
    acq = importlib.import_module('AL','AcquisitionFunctions')
    function = getattr(acq,config.un_function)

    trainer.pool_x = pool
    trainer.pool_y = None
    to_annotate = trainer.acquire(function,**params)
