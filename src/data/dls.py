""" helper functions for building DataLoaders """
import os
from pathlib import Path

from src.data.dblock import build_dblock


def build_dataloaders(data_path, bs, resize_sz = 256, norm = False, rand_seed = 144):
    """
    :param:
        data_path : str/ Path, path to wheat datasets
        resize_sz : int, length after resized (assume square)
        rand_seed : int, andom seed id
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)
        
    dblk = build_dblock(data_path, resize_sz, 
                        norm = norm, rand_seed = rand_seed)
    dls = dblk.dataloaders(data_path / 'train', bs = bs)
    dls.c = 2
    return dls