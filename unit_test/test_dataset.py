"""
inspect (image, target) with NO boxes or 1 boxes:
- 1 boxes only: 41457a646, 4e6c05213, 76595919e
- NO boxes: ffbb9c623, ccb3892c1
"""
import os
import time

import numpy as np
import pandas as pd

from src.data.datasets import get_datasets
from src.data.df_utils import read_boxes_df, get_kfolds_df


csv_path = '/userhome/34/h3509807/wheat-data/train.csv'
train_dir = '/userhome/34/h3509807/wheat-data/train'
boxes_df = read_boxes_df(csv_path)
kfolds_df = get_kfolds_df(boxes_df, k = 3)

fold_idx = 0
resize_sz = 512
is_cutmix = True
is_cutoff = False
train_ds, valid_ds = get_datasets(resize_sz, boxes_df, kfolds_df, 
                                  train_dir, fold_idx = fold_idx, 
                                  is_cutmix = is_cutmix, is_cutoff = is_cutoff)


# time sampling from datasets 
# without cutmix: transform iter = 10 not much different from transforms iter = 1
# with cutmix: also not much difference
def time_dataset(ds, iter_n = 25):
    start_t = time.time()

    for i in range(iter_n):
        img, target, img_id = ds[i]
        boxes_n, labels_n = target['boxes'].shape[0], target['labels'].shape[0]
        assert boxes_n == labels_n, f'boxes_n not align with labels_n'

    end_t = time.time()
    rate = (end_t - start_t) / iter_n
    print(f'rate over {iter_n} iterations: {rate} s')
    return None

for i in range(3):
    time_dataset(train_ds)


# sample with one boxes only
ids_w1box = ['41457a646', '4e6c05213', '76595919e']
outs = dict()

ids = kfolds_df[kfolds_df.fold != fold_idx].index.values
for i, img_id in enumerate(ids_w1box):
    id_idx = np.argwhere(ids == img_id)[0][0]
    img, target, _img_id = train_ds[id_idx]
    boxes_n, labels_n = target['boxes'].shape[0], target['labels'].shape[0]
    assert boxes_n == labels_n, f'boxes_n not align with labels_n'
    assert _img_id == img_id, f'id not aligned: {_img_id} v.s. {img_id}'
    outs[i] = (img, target, _img_id)
    

