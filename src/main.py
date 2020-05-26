""" implement main training loop """
import os
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


DATA_PATH = Path('/userhome/34/h3509807/wheat-data')
SAVE_DIR = Path('models')
SUFFIX_NAME = 'final_retinanet_512'
RESIZE_SZ = 512
TEST_MODE = False
RAND_SEED = 144

BS = 16
INIT_LR = 1e-4
INIT_EPOCH = 20

IS_FT = True
FT_LR = slice(1e-6, 5e-4)
FT_EPOCH = 20

RATIOS = [0.5, 1, 2.]
#SCALES = [1, 2**(-1/3), 2**(-2/3)]
SCALES = [1., 0.6, 0.3]


get_dls = partial(build_dataloaders, data_path = DATA_PATH, 
                  resize_sz = RESIZE_SZ, norm = True, 
                  rand_seed = RAND_SEED, test_mode = TEST_MODE)
dls = get_dls(bs = BS)


model = get_retinanet()
retinanet_loss = get_retinanet_loss(ratios = RATIOS, scales = SCALES)
mAP_meter = partial(mAP, img_size = RESIZE_SZ, 
                    ratios = RATIOS, scales = SCALES,
                    iou_thresholds = None, detect_threshold = 0.5, 
                    nms_threshold = 0.3)


save_cb = CheckpointCallback(SAVE_DIR)
learn = Learner(dls, model, 
                loss_func = retinanet_loss, 
                splitter = split_param_groups,
                cbs = [save_cb])
                #metrics = mAP_meter())


learn.freeze()
learn.fit_one_cycle(INIT_EPOCH, INIT_LR)
if IS_FT:
    learn.dls = get_dls(bs = BS)
    learn.unfreeze()
    learn.fit_one_cycle(FT_EPOCH, FT_LR)


model_path = SAVE_DIR / f'{SUFFIX_NAME}_model.pth'
learn.save(f'{SUFFIX_NAME}_learner')
torch.save(learn.model.state_dict(), model_path)
print('final checkpoints saved')


