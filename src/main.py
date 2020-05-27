""" implement main training loop """
import os
import argparse
from functools import partial
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import torch
from fastai2.vision.all import Learner

from src.data.dls import build_dataloaders
from src.model.model import get_retinanet, split_param_groups
from src.metrics.loss import get_retinanet_loss
from src.callback.core_cbs import CheckpointCallback
from src.metrics.mAP import mAP
from src.config.retinanet import config
from src.config.utils import update_config


def train_run(cfg):
    get_dls = partial(build_dataloaders, data_path = cfg.DATA_PATH,
                      resize_sz = cfg.RESIZE_SZ, norm = True,
                      rand_seed = cfg.RAND_SEED, test_mode = cfg.TEST_MODE)
    dls = get_dls(bs = cfg.BS)
    
    model = get_retinanet(cfg.ARCH, cfg.BIAS)
    retinanet_loss = get_retinanet_loss(gamma = cfg.GAMMA, alpha = cfg.ALPHA,
                                        ratios = cfg.RATIOS, scales = cfg.SCALES)
    
    mAP_meter = partial(mAP, img_size = cfg.RESIZE_SZ,
                        ratios = cfg.RATIOS, scales = cfg.SCALES,
                        iou_thresholds = None, 
                        detect_threshold = cfg.DETECT_THRESHOLD,
                        nms_threshold = self.NMS_THRESHOLD)
    save_cb = CheckpointCallback(cfg.PREFIX_NAME, 2)    
    
    learn = Learner(dls, model, 
                    path = './models', 
                    model_dir = cfg.MODEL_DIR,
                    loss_func = retinanet_loss, 
                    splitter = split_param_groups, 
                    cbs = [save_cb], metrics = mAP_meter())
    
    learn.freeze()
    learn.fit_one_cycle(cfg.INIT_EPOCH, cfg.INIT_LR)
    if cfg.IS_FT:
        learn.dls = get_dls(bs = cfg.BS)
        learn.unfreeze()
        learn.fit_one_cycle(cfg.FT_EPOCH, cfg.FT_LR)
    return None


if __name__ == '__main__':
    train_run(config)
