from pathlib import Path

import torch

from fastai2.vision.all import AvgMetric


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
                

