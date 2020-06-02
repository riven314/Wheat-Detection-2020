from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def get_efficientdet(resize_sz, backbone_ckpt):
    """ 
    backbone: tf_efficientdet_d5.pth 
    (aka efficientdet_d5-ef44aea8.pth/ tf_efficientnet_b5_ra-9a3e5369.pth) 
    
    backbone_ckpt
    e.g. ~/.cache/torch/checkpoints/tf_efficientnet_b5_ra-9a3e5369.pth
    """
    config = get_efficientdet_config('tf_efficientdet_d5')
    
    net = EfficientDet(config, pretrained_backbone = False)
    net.load_state_dict(torch.load(backbone_ckpt))
        
    config.num_classes = 1
    config.image_size = resize_sz
    net.class_net = HeadNet(config, num_outputs = config.num_classes, 
                            norm_kwargs = dict(eps = .001, momentum = .01))
    return DetBenchTrain(net, config)

