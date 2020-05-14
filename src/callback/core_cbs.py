from pathlib import Path

import torch

from fastai2.vision.all import patch, tensor
from fastai2.vision.all import TensorPoint
from fastai2.vision.all import AvgMetric
from fastai2.vision.all import Callback, TrainEvalCallback, Recorder


class FasterRCNNCallback(Callback):
    """ 
    essential changes for fastai2 Learner to interface with torchvision FasterRCNN 
    """
    def __init__(self, img_size):
        self.img_size = img_size
        
    def _decode_bboxs(self, enc_bboxs):
        sz = self.img_size
        return TensorPoint((enc_bboxs + 1)*tensor(256).float()/2, img_size = sz)
    
    def begin_batch(self):
        """ 
        re-arrange both self.learn.yb and self.learn.xb format
        self.learn.yb = [{'boxes': tensor(BS, 4), 'labels': tensor(BS)}, ...] 
        """
        # tupify, listify self.learn.xb
        self.learn.xb = ([x for x in self.learn.xb[0]], ) 
        
        # tupify, listify, dictionarize yb
        bboxs_b, cats_b = self.learn.yb
        
        yb = []
        for bboxs, cats in zip(bboxs_b, cats_b):
            idxs = torch.where(cats != 0)
            tmp_dict = {'boxes': self._decode_bboxs(bboxs[idxs]), 
                        'labels': cats[idxs]}
            yb.append(tmp_dict)
        
        self.learn.yb = (yb,)
        return None
    
    def after_fit(self):
        pass

    
class CheckpointCallback(Callback):
    """
    torch.save(model.state_dict(), PATH)
    
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    """
    def __init__(self, save_dir, save_period = 2):
        self.save_dir = save_dir if isinstance(save_dir, Path) else Path(save_dir)
        self.save_period = 2
        
    def after_epoch(self):
        if self.epoch % self.save_period == 0:
            model_path = self.save_dir / 'latest_model.pth'
            self.learn.save('latest_learner')
            torch.save(self.learn.model.state_dict(), model_path)
            print(f'latest checkpoints saved: {self.epoch}')
                
        
@patch
def begin_validate(self: TrainEvalCallback):
        """Set the model in train mode to output loss"""
        #print('monkey patched TrainEvalCallback.begin_validate')
        #self.model.eval()
        #print('begin validate, eval mode disabled!!!')
        self.learn.training = False
        

@patch
def after_batch(self: Recorder):
    """ 
    only enable model.eval mode on AvgMetric (i.e. mAP) calculation 
    this is a bad fix, assuming:
    1. AvgMetric is eval later than AvgLoss
    2. after Recorder callback, the rest of callback don't use self.learn.pred again
    """
    if len(self.yb) == 0: return
    mets = self._train_mets if self.training else self._valid_mets
    
    for met in mets: 
        if isinstance(met, AvgMetric):
            # overwrite self.learn.pre
            self.model.eval()
            self.learn.pred = self.model(*self.learn.xb)
        else:
            self.model.train()
        met.accumulate(self.learn)
    self.model.train()
    
    if not self.training: return
    self.lrs.append(self.opt.hypers[-1]['lr'])
    self.losses.append(self.smooth_loss.value)
    self.learn.smooth_loss = self.smooth_loss.value
