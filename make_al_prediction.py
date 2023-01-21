import numpy as np
import sys
import os
import re
import argparse
import pickle
import importlib

from make_tile_for_patch import pf_form
from make_alrun import global_trainer

from types import SimpleNamespace
from keras import backend as K

#AL System imports
from AL.Common import load_model_weights
from Utils.CacheManager import CacheManager

predictor = None

def execute_al_prediction(train,test,qa_config,proj_path,iteration):

    global predictor
    global global_trainer
    
    kwargs = {}
    #Data
    kwargs['data'] = "CellRep"
    kwargs['predst'] = qa_config.get("common","pool").rstrip('/')
    kwargs['testdir'] = qa_config.get("common","testset").rstrip('/')
    kwargs['split'] = (0.9,0.01,0.09)
    kwargs['tdim'] = (qa_config.getint("active_learning","input_size"),)*2
    kwargs['delay_load'] = True
    
    #Training
    kwargs['tnet'] = qa_config.get("active_learning","tnet")
    kwargs['network'] = qa_config.get("active_learning","net")
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
    kwargs['tnpred'] = 1
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
    x_test,y_test = ds.run_dir(config.testdir)

    print("Annotated test patches: {}".format(len(test[0])))
    print("Test patches available in test dir ({}): {}".format(config.testdir,len(x_test)))

    if not x_test is None and not y_test is None:
        test[0].extend(x_test)
        test[1].extend(y_test)

    if len(test[0]) == 0 or len(test[1]) == 0:
        return (0,0)
    
    if global_trainer is None:
        ts = importlib.import_module('Trainers',config.strategy)
        trainer = getattr(ts,config.strategy)(config)
    else:
        K.clear_session()
        trainer = global_trainer

    trainer.train_x = train[0][:-kwargs['val_set']]
    trainer.train_y = train[1][:-kwargs['val_set']]
    trainer.val_x = train[0][-kwargs['val_set']:]
    trainer.val_y = train[1][-kwargs['val_set']:]
    trainer.test_x = test[0]
    trainer.test_y = test[1]

    if predictor is None:
        predictor = trainer._build_predictor()
    else:
        K.clear_session()
        
    res = trainer.test_target(predictor,iteration,end_train=True,results=True)
    global_trainer = None
    K.clear_session()
    del(trainer)
    return res

