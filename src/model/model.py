import torch.nn as nn

from fastai2.vision.all import L
from fastai2.vision.all import resnet34, create_body, get_c, params

from src.model.RetinaNet import RetinaNet


def get_retinanet():
    encoder = create_body(resnet34, pretrained = True)
    arch = RetinaNet(encoder, 1, final_bias = -4)
    return arch


def split_param_groups(m):
    return L(m.encoder, 
              nn.Sequential(
                  m.c5top5, m.c5top6, m.p6top7, m.merges, 
                  m.smoothers, m.classifier, m.box_regressor)
            ).map(params)