import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from fastai2.vision.all import L
from fastai2.vision.all import resnet34, resnet50
from fastai2.vision.all import create_body, get_c, params

from src.model.RetinaNet import RetinaNet


def get_retinanet(arch = 'resnet34-imagenet', bias = -4):
    """
    :param:
        arch : str, resnet34-imagenet/ resnet50-imagenet/ resnet50-coco 
        bias : float, prior prob of foreground objects (before sigmoid)
    """
    if arch == 'resnet50-coco':
        print('resnet50-coco is selected')
        encoder = create_coco_resnet_body()
        
    elif arch == 'resnet34-imagenet':
        print('resnet34-imagenet is selected')
        backbone = resnet34
        encoder = create_body(backbone, pretrained = True)
        
    elif arch == 'resnet50-imagenet':
        print('resnet50-imagenet is selected')
        backbone = resnet50
        encoder = create_body(backbone, pretrained = True)
        
    arch = RetinaNet(encoder, 1, final_bias = bias) # class no. = 1
    return arch


def create_coco_resnet_body():
    """ only resnet50 available for COCO pretrained weight """
    fpn = fasterrcnn_resnet50_fpn(pretrained = True)
    # get resnet sub-modules
    for k, sub_modules in fpn.backbone.named_children():
        if k == 'body':
            break
    # re-construct sub-modules from nn.ModuleDict to nn.Sequential
    modules = []
    for name, module in sub_modules.items():
        modules += [module]
    return nn.Sequential(*modules)


def split_param_groups(m):
    return L(m.encoder, 
              nn.Sequential(
                  m.c5top5, m.c5top6, m.p6top7, m.merges, 
                  m.smoothers, m.classifier, m.box_regressor
                 )
            ).map(params)
