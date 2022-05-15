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
import openslide
import sys
import os
import time
import re
import multiprocessing
import argparse
from PIL import Image

pf_form = '(UN-(?P<unc>[0-9])+-){,1}(?P<tcga>TCGA-.*-.*-.*-.*-.*)-(?P<x>[0-9]+)-(?P<y>[0-9]+)-(?P<s1>[0-9]+)-(?P<s2>[0-9]+)(_(?P<lb>[01])){,1}\\.png'
wsi_form = '(?P<tcga>TCGA)-(?P<tss>[\\w]{2})-(?P<part>[\\w]{4})-(?P<sample>[\\d]{2}[A-Z]{0,1})-(?P<portion>[\\d]{2}[A-Z]{0,1})-(?P<plate>[\\w]{3}){0,1}'

def check_existing_tile(patch_name,tile_dest):

    if not os.path.isdir(tile_dest):
        os.mkdir(tile_dest)
        return None
    
    tiles = list(filter(lambda f:f.endswith('.png'),os.listdir(tile_dest)))
    rex = re.compile(pf_form)
    pm = rex.fullmatch(patch_name)
    if pm is None:
        print("Not a standard patch name: {}".format(patch_name))
        return None
    wsi = pm.group('tcga')
    px,py,ps1 = int(pm.group('x')),int(pm.group('y')),int(pm.group('s1'))

    for t in tiles:
        match = rex.fullmatch(t)
        if not match:
            print("Unknown file:{}".format(t))
            continue
        tw = match.group('tcga')
        ts1 = int(match.group('s1'))
        if tw == wsi:
            tx,ty = int(match.group('x')),int(match.group('y'))
            #Patch should be within 100 pixels from tile borders
            if (tx+100 < px <  (tx+ts1-(ps1+100))) and (ty+100 < py <  (ty+ts1-(ps1+100))):
                print(f"px: {px}; tx: {tx}; py: {py}; ty: {ty}")
                ts2 = int(match.group('s2'))
                return (t,*get_patch_position((tx,ty,ts1,ts2),(px,py,int(pm.group('s1')))))
                #return (t,px-tx,py-ty,ps1)

    return None

def get_patch_position(tile,patch):
    """
    tile: tuple -> (tile x position, tile y position, original tile size, output tile size)
    patch: tuple -> (patch x position, patch y position, patch size)
    """
    tx,ty,ts1,ts2 = tile
    px,py,ps = patch

    #Adjust coordinates to output tile size
    px = round((px-tx) * (ts2/ts1))
    py = round((py-ty) * (ts2/ts1))

    return (px,py,ps)

def make_tile(patch_name,wsi_dir,tile_size,tile_dest):
    """
    Should return the tile and the patch position within this tile.

    Return: tile name, patch position relative to tile, patch size in magnified tile (tile name,x,y,patch_size)
    """

    #First check if alreday exists a tile for this patch
    tile = check_existing_tile(patch_name,tile_dest)
    if not tile is None:
        return tile

    #Find WSI from which patch came from
    wrex = re.compile(wsi_form)
    wsis = list(filter(lambda f:f.endswith('.svs'),os.listdir(wsi_dir)))
    rex = re.compile(pf_form)
    pm = rex.fullmatch(patch_name)
    pwsi = pm.group('tcga')
    wsi = None
    
    for w in wsis:
        wm = wrex.match(w)
        wname = None
        if wm is None:
            print("Couldn't match file name: {}".format(w))
            return None
        else:
            wname = wm.group()
        if wname == pwsi:
            wsi = w

    if wsi is None:
        print("The Whole Slide Image from which patch {} come from is not available".format(patch_name))
        return None,0,0,0
    else:
        print("Extracting tile from {};".format(wsi))
    
    #Create tile, save and return it
    base_pw = 100
    amp = int(tile_size/base_pw)
    
    try:
        oslide = openslide.OpenSlide(os.path.join(wsi_dir,wsi));
        #mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
            mpp = float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
            mag = 10.0 / mpp
        elif "XResolution" in oslide.properties:
            mag = 10.0 / float(oslide.properties["XResolution"]);
        elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
            mag = 10.0 / float(oslide.properties["tiff.XResolution"]);
        else:
            mag = 10.0 / float(0.254);
    except:
        print('{}: exception caught'.format(slide_name));
        exit(1);

    pw = float(int(10 * base_pw * mag / 20)) / 10.0
    out_pw = int(base_pw * amp)
    pw_amp = int(pw * amp)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]    

    px,py = int(pm.group('x')),int(pm.group('y'))
    x = int(px+0.5*int(pm.group('s1'))) - int(pw_amp/2)
    y = int(py+0.5*int(pm.group('s1'))) - int(pw_amp/2)

    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    
    if x + pw_amp > width:
        x = x - ((x + pw_amp) - width) 

    if y + pw_amp > height:
        y = y - ((y + pw_amp) - height) 

    print("Making tile from position ({},{}) for patch ({},{})".format(x,y,px,py))
    patch = oslide.read_region((x, y), 0, (pw_amp, pw_amp));
    fname = '{}/{}-{}-{}-{}-{}.png'.format(tile_dest, pwsi, x, y, pw_amp, out_pw);
    patch = patch.resize((out_pw, out_pw), Image.ANTIALIAS);
    patch.save(fname);

    oslide.close()

    return (fname,*get_patch_position((x,y,pw_amp,tile_size),(px,py,int(pm.group('s1')))))


if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract tiles from a WSI \
        discarding background.')
        
    parser.add_argument('-wd', dest='wd', type=str,default='WSI', 
        help='Path to WSIs to tile (directory containing .svs images).')        
    parser.add_argument('-td', dest='tile_dir', type=str, default=None, 
        help='Save extracted tiles to this location.')
    parser.add_argument('-pd', dest='project_dir', type=str, default=None, 
        help='Patch location.')
    parser.add_argument('-mp', dest='mp', type=int, 
        help='Use multiprocessing. Number of processes to spawn', default=2,required=False)
    parser.add_argument('-label', action='store_true', dest='label',
        help='Generate labels for the patches from heatmaps.',default=False)
    parser.add_argument('-txt_label', action='store_true', dest='txt_label',
        help='Generate labels for the patches from heatmaps.',default=False)    
    parser.add_argument('-hm', dest='heatmap', type=str,default=None, 
        help='Heatmaps path.')
    parser.add_argument('-ts', dest='tile_size', type=int, 
        help='Tile size in 20x magnification (Default 500)', default=2000,required=False)
    parser.add_argument('-wr', dest='white', type=float, 
        help='Maximum white ratio allowed for each patch (Default: 0.20)', default=0.2,required=False)
    parser.add_argument('-db', action='store_true', dest='debug',
        help='Use to make extra checks on labels and conversions.',default=False)
    parser.add_argument('-hmc', action='store_true', dest='hmc',
        help='Check heatmap coordinates for duplicates.',default=False)
    parser.add_argument('-hmq', action='store_true', dest='hmq',
        help='Check heatmap file names only and exit.',default=False)
    
    config, unparsed = parser.parse_known_args()

    if config.project_dir is None:
        print("You should provide the path to project data")
        sys.exit(1)

    if config.tile_dir is None:
        config.tile_dir = os.path.join(config.project_dir,'tiles')
        
    if not os.path.exists(config.tile_dir):
        os.mkdir(config.tile_dir)

    if not os.path.isdir(config.wd):
        print("Path not found: {}".format(config.wd))
        sys.exit(1)
    
    patches = list(filter(lambda f:f.endswith('.png'),os.listdir(config.project_dir)))
    
    with multiprocessing.Pool(processes=config.mp) as pool:
        results = [pool.apply_async(make_tile,(i,config.wd,config.tile_size,config.tile_dir)) for i in patches]

        rt = [r.get() for r in results]
        
    print(rt)
