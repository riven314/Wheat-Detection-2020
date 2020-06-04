import os

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from src.data.df_utils import read_boxes_df, get_kfolds_df
from src.data.datasets import get_datasets


def get_dataloaders(resize_sz, csv_path, bs, kfolds = 5, fold_idx = 0, 
                    is_cutmix = False, is_cutoff = False, workers_n = 4):
    train_dir = f'{os.path.split(csv_path)[0]}/train'
    
    boxes_df = read_boxes_df(csv_path)
    kfolds_df = get_kfolds_df(boxes_df, kfolds = kfolds)
    
    train_ds, valid_ds = get_datasets(resize_sz, boxes_df, kfolds_df, train_dir, 
                                     fold_idx, is_cutmix, is_cutoff)
    
    train_dl = DataLoader(
        train_ds, batch_size = bs,
        num_workers = workers_n,
        shuffle = True,
        #sampler = RandomSampler(train_ds),
        pin_memory = True,
        collate_fn = collate_fn,
        drop_last = True
    )
    valid_dl = DataLoader(
        valid_ds, batch_size = bs,
        num_workers = workers_n,
        shuffle = False,
        sampler = SequentialSampler(valid_ds),
        pin_memory = True,
        collate_fn = collate_fn
    )
    return train_dl, valid_dl
    

def collate_fn(batch):
    return tuple(zip(*batch))