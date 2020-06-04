import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def read_boxes_df(csv_path):
    """ csv_path : .../train.csv """
    assert os.path.isfile(csv_path), f'csv not exist: {csv_path}'
    boxes_df = pd.read_csv(csv_path)

    boxes = np.stack(boxes_df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        boxes_df[column] = boxes[:,i]
    boxes_df.drop(columns=['bbox'], inplace=True)
    return boxes_df


def get_kfolds_df(boxes_df, kfolds = 5):
    """
    stratified by # boxes and data source of an image
    stratified means for every fold, stratify_group's distribution is the same
    ** currently, it only include images withn non-zero boxes
    
    :param:
        boxes_df : DataFrame, one row = one box
    :return:
        kfolds_df : DataFrame, one row = one image
    """
    skf = StratifiedKFold(n_splits = kfolds, shuffle = True, random_state = 42)

    kfolds_df = boxes_df[['image_id']].copy()
    kfolds_df.loc[:, 'boxes_count'] = 1
    kfolds_df = kfolds_df.groupby('image_id').count() # row count per image_id
    kfolds_df.loc[:, 'source'] = boxes_df[['image_id', 'source']].groupby('image_id').min()['source']
    kfolds_df.loc[:, 'stratify_group'] = np.char.add(
        kfolds_df['source'].values.astype(str),
        kfolds_df['boxes_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    kfolds_df.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X = kfolds_df.index, y = kfolds_df['stratify_group'])):
        kfolds_df.loc[kfolds_df.iloc[val_index].index, 'fold'] = fold_number
    return kfolds_df