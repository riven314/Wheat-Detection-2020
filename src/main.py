import os

from src.data.dls import build_dataloaders
from src.model.FasterRCNN import get_faster_rcnn, split_faster_rcnn_params
from src.callback.learner import WheatLearner


DATA_PATH = '/userhome/34/h3509807/wheat-data'

learn = WheatLearner(dls, model, 
                     loss_func = multi_loss, 
                     splitter = _faster_rcnn_split,
                     #metrics = map_getter,
                     cbs = EssentialCallback(img_size = 256))