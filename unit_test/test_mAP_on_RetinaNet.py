import os
import sys
import warnings 
warnings.filterwarnings("ignore")
path = os.path.join(os.getcwd(), '..')
sys.path.append(path)

import torch

from fastai2.vision.all import *

from src.data.dblock import build_dblock
from src.data.dls import build_dataloaders
from src.model.model import get_retinanet, split_param_groups
from src.metrics.loss import get_retinanet_loss
from src.metrics.mAP import mAP

data_path = '/userhome/34/h3509807/wheat-data'
bs = 8
resize_sz = 256
rand_seed = 144
norm = True
test_mode = True
lr = 1e-4
epochs = 3


# prepare learner, model, dataloaders
dls = build_dataloaders(
    data_path, bs = bs, resize_sz = resize_sz, 
    norm = norm, rand_seed = rand_seed, test_mode = test_mode
    )
model = get_retinanet()
retinanet_loss = get_retinanet_loss(ratios = None, scales = None)
learn = Learner(dls, model, 
                loss_func = retinanet_loss, 
                splitter = split_param_groups)
learn.freeze()

learn_path = 'models/retina_learner.pth'
if os.path.isfile(learn_path):
    print(f'learner weight found, load it in: {learn_path}')
    learn.load('retina_learner')
else:
    print('no learner weight, train a new one')
    learn.fit_one_cycle(epochs, lr)
    learn.save('retina_learner')

    
# transform model outputs to mAP expected inputs
b = learn.dls.one_batch()
b_imgs, b_bboxs_gts, b_clas_gts = b
with torch.no_grad():
    b_preds = learn.model(b_imgs)
b_clas_preds, b_bboxs_preds, sizes = b_preds


# take out one batch
clas_preds, bboxs_preds = b_clas_preds[0], b_bboxs_preds[0]
clas_gts, bboxs_gts = b_clas_gts[0], b_bboxs_gts[0]
clas_preds = clas_preds.cpu().numpy().squeeze()
bboxs_preds = bboxs_preds.cpu().numpy()
clas_gts = clas_gts.cpu().numpy()
bboxs_gts = bboxs_gts.cpu().numpy()


# massage one batch of predictions (filter high confidence + sort by descending order)
thre_idxs = np.argwhere(clas_preds >= 0.5).squeeze()
thre_clas_preds = clas_preds[thre_idxs]
thre_bboxs_preds = bboxs_preds[thre_idxs]

sort_idxs = np.argsort(-thre_clas_preds)
sort_clas_preds = thre_clas_preds[sort_idxs]
sort_bboxs_preds = thre_bboxs_preds[sort_idxs]
assert sort_clas_preds.shape[0] == sort_bboxs_preds.shape[0]


# massage one batch of ground-truths
valid_idxs = np.argwhere(clas_gts == 1).squeeze()
valid_clas_gts = clas_gts[valid_idxs]
valid_bboxs_gts = bboxs_gts[valid_idxs]
assert valid_bboxs_gts.shape[0] == valid_clas_gts.shape[0]


# massage bbox to [x, y, w, h]


