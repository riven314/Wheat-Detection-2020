import os
import json
from pathlib import Path

import torch
from stuf import stuf

from fastai2.vision.all import AvgMetric
from fastai2.vision.all import Callback


class RetinaNetCallback(Callback):
    """ 
    essential changes for fastai2 Learner to interface with torchvision FasterRCNN 
    """
    def __init__(self):
        pass
    
    def begin_batch(self):
        pass
    
    def after_fit(self):
        pass

    
class ConfigCallback(Callback):
    """ save config and last metrics as a json file """
    def __init__(self, config):
        self.config = config.copy()
    
    def begin_fit(self):
        self.save_dir = self.learn.path / self.learn.model_dir
        os.makedirs(str(self.save_dir), exist_ok = True)
        
    def after_fit(self):
        self.config.FINAL_RECORD = list(self.learn.final_record)
        json_path = str(self.save_dir / 'config.json')
        self.config.DATA_PATH = str(self.config.DATA_PATH)
        self.config.MODEL_DIR = str(self.config.MODEL_DIR)
        self.config.FT_LR = [self.config.FT_LR.start, self.config.FT_LR.stop]
        with open(json_path, 'w') as f:
            json.dump(self.config, f, indent = 2)
        print(f'config file written: {json_path}')
    
    
class CheckpointCallback(Callback):
    """
    torch.save(model.state_dict(), PATH)
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    :param:
        prefix_final : str, prefix name of the final saved model/learner
        save_period : int, period of epochs for saving model/learner
    """
    def __init__(self, prefix_final = 'final', save_period = 2):
        self.save_period = 2
        self.prefix_final = prefix_final
    
    def _checkpoint(self, prefix):
        model_path = self.save_dir / f'{prefix}_model.pth'
        torch.save(self.learn.model.state_dict(), model_path)
        self.learn.save(f'{prefix}_learner')
        print(f'latest checkpoints saved: {self.epoch}')
        
    def begin_fit(self):
        self.save_dir = self.learn.path / self.learn.model_dir
        os.makedirs(str(self.save_dir), exist_ok = True)
        
    def after_epoch(self):
        if self.epoch % self.save_period == 0:
            self._checkpoint(prefix = 'latest')
            
    def after_fit(self):
        self._checkpoint(prefix = self.prefix_final)

