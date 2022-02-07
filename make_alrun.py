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

global_trainer = None

def run_active_learning(pool,spool,acq_idx,data,qa_config,proj_path,iteration):

    global global_trainer
    
    kwargs = {}
    #Data
    kwargs['data'] = "CellRep"
    kwargs['predst'] = qa_config.get("common","pool")
    kwargs['split'] = (0.9,0.01,0.09)
    kwargs['tdim'] = (qa_config.getint("active_learning","input_size"),)*2
    kwargs['delay_load'] = True
    
    #Training
    kwargs['tnet'] = qa_config.get("active_learning","tnet")
    kwargs['net'] = qa_config.get("active_learning","net")
    kwargs['batch_size'] = qa_config.getint("train_ae","batchsize")
    kwargs['learn_r'] = 0.0005
    kwargs['epochs'] = qa_config.getint("active_learning","alepochs")
    kwargs['train_set'] = qa_config.getint("active_learning","initial_set")
    kwargs['val_set'] = qa_config.getint("active_learning","val_size")
    kwargs['weights_path'] = os.path.join(proj_path,'models')
    kwargs['new_net'] = True
    kwargs['bdir'] = os.path.join(proj_path,'cache')
    kwargs['cache'] = os.path.join(proj_path,'cache')
    kwargs['logdir'] = os.path.join(proj_path,'cache')
    kwargs['augment'] = False
    kwargs['batch_norm'] = False
    kwargs['model_path'] = os.path.join(proj_path,'models')
    kwargs['save_dt'] = False
    kwargs['save_w'] = False
    kwargs['plw'] = False
    kwargs['save_var'] = False
    kwargs['f1period'] = 0
    kwargs['lyf'] = 0
    
    #AL
    kwargs['phi'] = qa_config.getint("active_learning","phi")
    kwargs['tnphi'] = qa_config.getint("active_learning","tnphi")
    kwargs['strategy'] = qa_config.get("active_learning","strategy")
    kwargs['keepimg'] = True
    kwargs['spool'] = 1
    kwargs['spool_f'] = None
    kwargs['acquire'] = qa_config.getint("active_learning","aq_size")
    kwargs['clusters'] = 20
    kwargs['ffeat'] = None
    kwargs['recluster'] = 0
    kwargs['ac_function'] = 'dada'
    kwargs['un_function'] = qa_config.get("active_learning","un_function")
    kwargs['emodels'] = qa_config.getint("active_learning","emodels")
    kwargs['dropout_steps'] = qa_config.getint("active_learning","dropout_steps")
    kwargs['sample'] = qa_config.getint("active_learning","subpool")
    
    #System
    kwargs['gpu_count'] = qa_config.getint("cuda","gpucount")
    kwargs['cpu_count'] = qa_config.getint("pooling","npoolthread")
    kwargs['verbose'] = qa_config.getint("common","verbose")
    kwargs['progressbar'] = False
    kwargs['debug'] = False
    kwargs['pred_size'] = 15000
    kwargs['info'] = True

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
        'tiles.pik':os.path.join(config.cache,'tiles.pik'),
        'test_pred.pik':os.path.join(config.logdir,'test_pred.pik'),
        'cae_model.h5':os.path.join(config.model_path,'cae_model.h5'),
        'vgg16_weights_notop.h5':os.path.join('PretrainedModels','vgg16_weights_notop.h5')}

    cache_m = CacheManager(locations=files)
    
    dsm = importlib.import_module('Datasources',config.data)
    ds = getattr(dsm,config.data)(config.predst,config.keepimg,config)

    Y = None
    if global_trainer is None:
        ts = importlib.import_module('Trainers',config.strategy)
        trainer = getattr(ts,config.strategy)(config)
        trainer.train_x = data[0][:kwargs['val_set']]
        trainer.train_y = data[1][:kwargs['val_set']]
        trainer.val_x = data[0][kwargs['val_set']:]
        trainer.val_y = data[1][kwargs['val_set']:]
        trainer.pool_size = qa_config.getint("active_learning","subpool")
        Y = np.zeros(shape=spool.shape,dtype=np.int8)
        trainer.superp_x = spool
        trainer.superp_y = Y

        if not pool is None
            print("Using cached subpool: {} patches".format(len(pool)))
            trainer.pool_x = pool
            trainer.pool_y = Y
            trainer.acq_idx = acq_idx
        elif config.sample != 1.0:
            X,Y,idx = ds.sample_metadata(config.sample,data=(trainer.superp_x,Y))
            trainer.sample_idx = idx
            print("Generating new subpool: {} patches".format(len(X)))
            trainer.pool_x = X
            trainer.pool_y = Y
        else:
            trainer.pool_x = spool
            trainer.pool_y = Y
            
    else:
        trainer = global_trainer

    params = {}
    
    model = trainer.load_modules(config.net,ds)
    model.setPhi(config.phi)
    params['model'] = model
    params['acquisition'] = iteration
    params['config'] = config
    tmodel,sw_thread,_ = trainer._target_net_train(model)
    params['sw_thread'] = sw_thread[0] if len(sw_thread) == 1 else sw_thread
    params['emodels'] = tmodel
    params['return_aq'] = True
    params['single_r'] = False
    
    acq = importlib.import_module('AL','AcquisitionFunctions')
    function = getattr(acq,config.un_function)

    to_annotate,pool = trainer.acquire(function,**params)
        
    print(to_annotate)

    return (to_annotate,pool,trainer.superp_x,trainer.acq_idx)
