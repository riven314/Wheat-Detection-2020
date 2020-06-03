import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(resize_sz, is_train, is_cutoff = False):
    """ is_cutoff is used only when is_train = True """
    bbox_params = A.BboxParams(
            format = 'pascal_voc',
            min_area = 0, 
            min_visibility = 0,
            label_fields = ['labels']
        )
    
    if is_train and is_cutoff:
        tfms = [
            A.Resize(height=resize_sz, width=resize_sz, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0)
        ]
    # valid / train wo cutoff augmentation
    else:
        tfms = [
            A.Resize(height=resize_sz, width=resize_sz, p=1),
            ToTensorV2(p=1.0)
        ]
        
    if is_train:
        add_tfms = [ # height, width: after crop and resize
            A.RandomSizedCrop(
                min_max_height = (800, 800), 
                height = 1024, width = 1024, p = 0.5
            ),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit = 0.2, sat_shift_limit = 0.2, 
                    val_shift_limit = 0.2, p = 0.9
                ),
                A.RandomBrightnessContrast(
                    brightness_limit = 0.2, 
                    contrast_limit = 0.2, p = 0.9
                ),
            ], p = 0.9),
            A.ToGray(p = 0.01),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
        ]
        tfms = add_tfms + tfms
        
    return A.Compose(tfms, p = 1., bbox_params = bbox_params)