import torch.nn as nn

from fastai2.vision.all import L
from fastai2.vision.all import resnet34, resnet50
from fastai2.vision.all import create_body, get_c, params

from src.model.RetinaNet import RetinaNet


def get_retinanet(arch = 'resnet34', bias = -4):
    if arch == 'resnet34':
        backbone = resnet34
    elif arch == 'resnet50':
        backbone = resnet50
    encoder = create_body(backbone, pretrained = True)
    arch = RetinaNet(encoder, 1, final_bias = bias) # class no. = 1
    return arch


def split_param_groups(m):
    return L(m.encoder, 
              nn.Sequential(
                  m.c5top5, m.c5top6, m.p6top7, m.merges, 
                  m.smoothers, m.classifier, m.box_regressor
                 )
            ).map(params)
