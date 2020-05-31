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
                    train_bn = True,
                    splitter = split_param_groups, 
                    cbs = [save_cb, cfg_cb], 
                    metrics = mAP_meter())
    
    learn.freeze()
    learn.fit_one_cycle(cfg.INIT_EPOCH, cfg.INIT_LR)
    if cfg.IS_FT:
        learn.dls = get_dls(bs = cfg.BS)
        learn.unfreeze()
        learn.fit_one_cycle(cfg.FT_EPOCH, cfg.FT_LR)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_DIR', type = str, required = True, help = 'dir name for save models')
    parser.add_argument('--ARCH', type = str, required = True, help = 'resnet50-coco/ resnet34-imagenet/ resnet50-imagenet')
    parser.add_argument('--RESIZE_SZ', type = int, default = 256)
    parser.add_argument('--BS', type = int, default = 32)
    parser.add_argument('--INIT_LR', type = float, default = 1e-4)
    parser.add_argument('--INIT_EPOCH', type = int, required = True)
    parser.add_argument('--IS_FT', action = 'store_true')
    parser.add_argument('--FT_EPOCH', type = int)
    parser.add_argument('--ALPHA', type = float, default = 0.5)
    parser.add_argument('--GAMMA', type = int, default = 1)
    parser.add_argument('--NMS_THRESHOLD', type = float, default = 0.3)
    parser.add_argument('--BIAS', type = float, default = -2.)
    args = parser.parse_args()
    
    config = update_config(config, args)
    train_run(config)
