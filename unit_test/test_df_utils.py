import os

from src.data.df_utils import read_boxes_df, get_kfolds_df


csv_path = '/userhome/34/h3509807/wheat-data/train.csv'

boxes_df = read_boxes_df(csv_path)

# any oversampling for stratified purpose?
# what image has no boxes?
kfolds_df = get_kfolds_df(boxes_df, k = 5)