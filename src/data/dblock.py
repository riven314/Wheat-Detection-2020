""" helper functions for constructing DataBlcok """
import os

from src.data.utils import get_img_ids, decode_coco_json

from fastai2.vision.all import DataBlock

from fastai2.vision.all import imagenet_stats
from fastai2.vision.all import ImageBlock, BBoxBlock, BBoxLblBlock
from fastai2.vision.all import RandomSplitter
from fastai2.vision.all import Resize, Rotate, Flip, Dihedral, Normalize


def build_dblock(data_path, resize_sz, rand_seed = 144):
    json_path = data_path / 'train.json'
    _, _, img2bbox = decode_coco_json(json_path)
    
    blks = (ImageBlock, BBoxBlock, BBoxLblBlock)
    get_ids_func = get_img_ids(json_path)
    getters_func = [lambda o: data_path / 'train' / o, 
                    lambda o: img2bbox[o][0], 
                    lambda o: img2bbox[o][1]]
    
    rand_splitter = RandomSplitter(valid_pct = 0.2, seed = rand_seed)
    item_tfms = [Resize(resize_sz)]
    batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize.from_stats(*imagenet_stats)]
    
    dblock = DataBlock(
        blocks = blks, splitter = rand_splitter,
        get_items = get_ids_func, getters = getters_func,
        item_tfms = item_tfms, batch_tfms = batch_tfms, n_inp = 1
        )
    return dblock
                       