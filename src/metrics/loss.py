import torch.nn as nn


class WeightedMultiLoss(nn.Module):
    """ dummy loss function for torchvision FasterRCNN """
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, tgts, **kwargs):
        loss = 0.
        for _, ind_loss in preds.items():
            loss += ind_loss
        return loss