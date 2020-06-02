import os
import random

import cv2
import numpy as np
import torch


def get_device():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return dev

    
def seed_everything(seed):
    """ https://pytorch.org/docs/stable/notes/randomness.html """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def read_image(image_id, train_dir):
    img_path = f'{train_dir}/{image_id}.jpg'
    assert os.path.isfile(img_path), f'image not exist: {img_path}'
    img = cv2.imread(f'{train_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img / 255.


def xywh2ltrb(bboxs):
    """ bboxs : np.array (N, 4) """
    ltrb_bboxs = bboxs.copy()
    ltrb_bboxs[:, 2] = bboxs[:, 0] + bboxs[:, 2] # x0 + w
    ltrb_bboxs[:, 3] = bboxs[:, 1] + bboxs[:, 3] # y0 + h
    return ltrb_bboxs


def ltrb2xywh(bboxs):
    """ bboxs : np.array (N, 4) """
    xywh_bboxs = bboxs.copy()
    ltrb_bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0] # x0 + w
    ltrb_bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1] # y0 + h
    return ltrb_bboxs