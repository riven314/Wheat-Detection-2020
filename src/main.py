import os
from functools import partial
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch

from src.data.dls import build_dataloaders
from src.model.FasterRCNN import get_faster_rcnn, split_faster_rcnn_params
from src.metrics.loss import WeightedMultiLoss
from src.metrics.mAP import calculate_batch_precision
from src.callback.core_cbs import FasterRCNNCallback, CheckpointCallback
from src.callback.core_cbs import begin_validate, after_batch
from src.callback.learner import WheatLearner


DATA_PATH = Path('/userhome/34/h3509807/wheat-data')
SAVE_DIR = Path('models')
RESIZE_SZ = 256
TEST_MODE = False
RAND_SEED = 144

BS = 8 # 16 backbone fine-tune OOM
INIT_LR = 2e-4
INIT_EPOCH = 30

IS_FT = True
FT_LR = slice(2e-6, 1e-4)
FT_EPOCH = 30


dls = build_dataloaders(DATA_PATH, BS, RESIZE_SZ, 
                        norm = False, rand_seed = RAND_SEED, 
                        test_mode = TEST_MODE)
model = get_faster_rcnn()

multi_loss = WeightedMultiLoss()
thresholds = [i for i in map(lambda i: i/100, range(50, 80, 5))]
mAP_calc = partial(calculate_batch_precision, thresholds = thresholds)

faster_rcnn_cb = FasterRCNNCallback(img_size = RESIZE_SZ)
save_cb = CheckpointCallback(SAVE_DIR)

learn = WheatLearner(dls, model, loss_func = multi_loss, 
                     splitter = split_faster_rcnn_params,
                     metrics = mAP_calc,
                     cbs = [faster_rcnn_cb, save_cb])
                     #metrics = map_getter,

learn.freeze()
learn.fit_one_cycle(INIT_EPOCH, INIT_LR)
if IS_FT:
    learn.unfreeze()
    learn.fit_one_cycle(FT_EPOCH, FT_LR)

model_path = SAVE_DIR / 'final_model.pth'
learn.save('final_learner')
torch.save(learn.model.state_dict(), model_path)
print('final checkpoints saved')


