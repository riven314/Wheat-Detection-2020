import os
import time

from src.data.dataloaders import get_dataloaders

resize_sz = 512
csv_path = '/userhome/34/h3509807/wheat-data/train.csv'
bs = 4
kfolds, fold_idx = 5, 0

train_dl, valid_dl = get_dataloaders(resize_sz, csv_path, bs, 
                                     kfolds = kfolds, fold_idx = fold_idx)


def time_dataloader(dl, iter_n = 15):
    start_t = time.time()

    for i, data in enumerate(dl):
        if i == 8:
            break

    end_t = time.time()
    rate = (end_t - start_t) / iter_n
    print(f'rate over {iter_n} iterations: {rate} s')
    return None


time_dataloader(train_dl, iter_n = 15)