import os

from fastai2.vision.all import noop

from src.data.utils import get_img_ids


def test_get_img_ids():
    json_path = '/userhome/34/h3509807/wheat-data/train.json'
    f = get_img_ids(json_path)
    img_ids = f(noop)
    return img_ids, f


if __name__ == '__main__':
    img_ids, f = test_get_img_ids()