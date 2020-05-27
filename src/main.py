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
from src.callback.core_cbs import CheckpointCallback, ConfigCallback
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
                        nms_threshold = cfg.NMS_THRESHOLD)
    save_cb = CheckpointCallback(cfg.PREFIX_NAME, 2)    
    cfg_cb = ConfigCallback(cfg)
    
    learn = Learner(dls, model, path = Path('./models'), 
                    model_dir = cfg.MODEL_DIR,
                    loss_func = retinanet_loss, 
                    splitter = split_param_groups, 
                    cbs = [save_cb, cfg_cb], 
                    metrics = mAP_meter())
    
    learn.freeze()
    learn.fit_one_cycle(cfg.INIT_EPOCH, cfg.INIT_LR)
    if cfg.IS_FT:
        learn.dls = get_dls(bs = cfg.BS)
        learn.unfreeze()
        learn.fit_one_cycle(cfg.FT_EPOCH, cfg.FT_LR)
    return learn


if __name__ == '__main__':
#     for bias in [-4., -2., -1., 0.]:
#         for gamma in [1, 2, 3]:
#             for alpha in [0.25, 0.5, 0.75]:
#                 for nms in [0.3, 0.4, 0.5]:
    for bias in [-4.]:
        for gamma in [2]:
            for alpha in [0.5]:
                for nms in [0.4]:
                    config.BIAS = bias
                    config.GAMMA = gamma
                    config.ALPHA = alpha
                    config.NMS_THRESHOLD = nms
                    config.MODEL_DIR = f'bias{bias}_gamma{gamma}_alpha{alpha}_nms{nms}'
                    print(f'training {config.MODEL_DIR}')
                    try:
                        learn = train_run(config)
                    except:
                        print(f'error skipped: {config.MODEL_DIR}')
                        continue
