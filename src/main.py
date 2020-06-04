import os

import torch

from src.config.GlobalConfig import GlobalConfig
from src.common.utils import get_device
from src.data.dataloaders import get_dataloaders
from src.model.efficientdet import get_efficientdet
from src.learner.Learner import Learner


backbone_ckpt = '/userhome/34/h3509807/.cache/torch/checkpoints/tf_efficientdet_d5-ef44aea8.pth'
csv_path = '/userhome/34/h3509807/wheat-data/train.csv'
resize_sz = 256
bs = 8
kfolds = 5
fold_idx = 0
is_cutmix = False
is_cutoff = True


device = get_device()
train_dl, valid_dl = get_dataloaders(resize_sz, csv_path, bs, 
                                     kfolds = kfolds, fold_idx = fold_idx,
                                     is_cutmix = is_cutmix, is_cutoff = is_cutoff)
model = get_efficientdet(resize_sz, backbone_ckpt)

learner = Learner(model, device, GlobalConfig)
learner.fit(train_dl, valid_dl)