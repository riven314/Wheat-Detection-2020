import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_faster_rcnn():
    """
    pretrain normalization is done within model
    
    i.e. 
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    """
    cls_n = 2
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cls_n)
    return model
    

def split_faster_rcnn_params(m):
    return L(m.backbone, m.rpn, m.roi_heads).map(params)