import os
import re

from fastai2.vision.all import noop, get_annotations


def parse_bbox_str(x):
    """ 
    parse one bbox info from train.csv
    e.g. '52.0 25.0 10.0 10.0' = [x, y, w, h] 
    """
    bbox = re.findall('([0-9]+[.]?[0-9]*)', x)
    #return [i for i in map(float, bbox)]
    return [i for i in map(float, bbox)]


def decode_coco_json(json_path):
    """
    read COCO-format json and extract info of all training samples
    e.g. image ids, and bbox mapping
    """
    assert os.path.isfile(json_path), f'json not exist: {json_path}'
    img_ids, lbl_bbox = get_annotations(json_path)
    img2bbox = dict(zip(imgs, lbl_bbox))
    return img_ids, lbl_bbox, img2bbox

    
def get_img_ids(json_path):
    def inner(noop):
        img_ids, lbl_bbox = get_annotations(json_path)
        return img_ids
    return inner
    
