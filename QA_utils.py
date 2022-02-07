import os
import pickle
import torch
import numpy as np
from QA_config import config

################################################################################
# Output either True or False if cuda is available for deep learning computations.
def has_cuda():
    # is there even cuda available?
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        # we require cuda version >=3.5
        capabilities = torch.cuda.get_device_capability(torch.cuda.current_device())
        major_version = capabilities[0]
        minor_version = capabilities[1]
        if major_version < 3 or (major_version == 3 and minor_version < 5):
            has_cuda = False
    print(f'Has cuda = {has_cuda}')
    return has_cuda


# Output a torch device to use.
def get_torch_device(gpuid = None):
    # Output a torch device with a preferred gpuid (use -2 to force cpu)
    if not gpuid:
        gpuid = config.getint("cuda","gpuid", fallback=0)

    device = torch.device(gpuid if gpuid != -2 and torch.cuda.is_available() else 'cpu')
    return device

################################################################################


################################################################################
# similar to the unix tail command - from https://stackoverflow.com/a/136368
def get_file_tail(file_path, lines=20):
    f = open(file_path, 'rb')
    total_lines_wanted = lines
    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = []
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            f.seek(0,0)
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count(b'\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = b''.join(reversed(blocks))
    return b'\n'.join(all_read_text.splitlines()[-total_lines_wanted:])

################################################################################

def tile_for_patch(patch):
    from make_tile_for_patch import make_tile
    
    wsidir = config.get('common','wsis',fallback='.')

    if not os.path.isdir(wsidir):
        return None,0,0,0
        
    tile_size = config.getint('common','tilesize',fallback=2000)
    tile_dest,patch_name = os.path.split(patch)

    return make_tile(patch_name,wsidir,tile_size,tile_dest)


################################################################################
def get_initial_train(cache):
    """
    Returns a tuple (train set, val set) val set can be None
    """
    from make_initial_trainset import generate_set
    
    return generate_set(path=config.get('common','pool'),
                            n=config.getint('active_learning','initial_set'),
                            nval=config.getint('active_learning','val_size'),
                            cache=cache)

################################################################################
def get_img_metadata(path):
    from make_initial_trainset import makeImg
    
    return makeImg(path)

################################################################################
def get_metadata_pool(cache,sp=False):
    if sp:
        pool_file = os.path.join(cache,'spool.pik')
    else:
        pool_file = os.path.join(cache,'pool.pik')
    pool = None
    if os.path.isfile(pool_file):
        with open(pool_file,'rb') as fd:
            pool = pickle.load(fd)
    else:
        print("Something is wrong, no {} file present".format(pool_file))
        return None
    
    return np.asarray(pool)

################################################################################
def save_updated_pool(cache,pool,sp=False):
    if sp:
        pool_file = os.path.join(cache,'spool.pik')
    else:
        pool_file = os.path.join(cache,'pool.pik')
    with open(pool_file,'wb') as fd:
        pickle.dump(pool,fd)

################################################################################
def save_update_idx(cache,acq_idx):
    acq_file = os.path.join(cache,'acq_idx.pik')

    with open(acq_file,'wb') as fd:
        pickle.dump(acq_idx,fd)
        
################################################################################
def get_metadata_acqidx(cache):
    acq_file = os.path.join(cache,'acq_idx.pik')
    acq = None

    if not os.path.isfile(acq_file):
        return None
    
    with open(acq_file,'rb') as fd:
        acq = pickle.load(fd)
    return acq

################################################################################

def run_al(proj_path,rois,config,iteration):
    from make_initial_trainset import makeImg
    from make_alrun import run_active_learning
    
    cache = os.path.join(proj_path,'cache')
    pool = get_metadata_pool(cache)
    spool = get_metadata_pool(cache,sp=True)
    acq_idx = get_metadata_acqidx(cache)
    train_x, train_y = [],[]

    for r in rois:
        train_x.append(makeImg(r.alpath))
        train_y.append(r.anclass)
    
    if not spool is None:
        print("Superpool size: {}".format(len(spool)))
        sel,pool,spool,acq_idx = run_active_learning(pool,spool,acq_idx,(train_x,train_y),config,proj_path,iteration)
        save_updated_pool(cache,pool)
        save_updated_pool(cache,spool,True)
        save_update_idx(cache,acq_idx)
        print("Updated pool size: {}".format(len(pool)))
        print("Superpool size: {}".format(len(spool)))
        del(pool)
        del(spool)
        return sel
    else:
        return None
    
