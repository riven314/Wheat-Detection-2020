import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch

from src.data.dls import build_dataloaders
from src.model.FasterRCNN import get_faster_rcnn, split_faster_rcnn_params
from src.metrics.loss import WeightedMultiLoss
from src.callback.core_cbs import FasterRCNNCallback, CheckpointCallback
from src.callback.core_cbs import begin_validate
from src.callback.learner import WheatLearner


DATA_PATH = Path('/userhome/34/h3509807/wheat-data')
SAVE_DIR = Path('models')
RESIZE_SZ = 256
TEST_MODE = False
BS = 16
LR = 2e-4
INIT_EPOCH = 10
FT_EPOCH = 20

dls = build_dataloaders(DATA_PATH, BS, RESIZE_SZ, 
                        norm = False, rand_seed = 144, 
                        test_mode = TEST_MODE)
model = get_faster_rcnn()
multi_loss = WeightedMultiLoss()
faster_rcnn_cb = FasterRCNNCallback(img_size = RESIZE_SZ)
save_cb = CheckpointCallback(SAVE_DIR)

learn = WheatLearner(dls, model, loss_func = multi_loss, 
                     splitter = split_faster_rcnn_params,
                     cbs = [faster_rcnn_cb, save_cb])
                     #metrics = map_getter,

learn.freeze()
learn.fit_one_cycle(INIT_EPOCH, LR)
learn.freeze_to(-2)
learn.fit_one_cycle(FT_EPOCH, LR / 10)

model_path = SAVE_DIR / 'final_model.pth'
learn.save('final_learner')
torch.save(learn.model.state_dict(), model_path)
print('final checkpoints saved')


