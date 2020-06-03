import os
from random import random, uniform, randint

import torch
import numpy as np
from torch.utils.data import Dataset

from src.common.utils import read_image, xywh2ltrb
from src.data.tfms import get_transforms


def get_datasets(resize_sz, boxes_df, kfolds_df, train_dir, 
                 fold_idx = 0, is_cutmix = False, is_cutoff = False):
    """
    :param: 
        fold_idx : fold index for validation set
        is_cutmix : bool, whether enable cutmix in Dataset (train set)
        is_cutoff : bool, whether enable small cutoff in transforms (train set)
    """
    train_ids = kfolds_df[kfolds_df.fold != fold_idx].index.values
    valid_ids = kfolds_df[kfolds_df.fold == fold_idx].index.values
    
    train_tfms = get_transforms(resize_sz, is_train = True, is_cutoff = is_cutoff)
    valid_tfms = get_transforms(resize_sz, is_train = False, is_cutoff = False)
    
    train_ds = DatasetRetriever(train_ids, boxes_df, train_dir, 
                                train_tfms, is_cutmix = is_cutmix)
    valid_ds = DatasetRetriever(valid_ids, boxes_df, train_dir, 
                                valid_tfms, is_cutmix = False)
    return train_ds, valid_ds


class DatasetRetriever(Dataset):
    def __init__(self, image_ids, boxes_df, train_dir, transforms = None, is_cutmix = False):
        super().__init__()

        self.image_ids = image_ids
        self.boxes_df = boxes_df
        self.transforms = transforms
        self.is_cutmix = is_cutmix
        self.train_dir = train_dir

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        # cutmix disabled in test mode 
        if self.is_cutmix:
            do_cutmix = random() > 0.5
            image, boxes = self.load_cutmix_image_and_boxes(index) if do_cutmix else self.load_image_and_boxes(index)
        else:
            image, boxes = self.load_image_and_boxes(index)
        
        # only one class
        labels = torch.ones((boxes.shape[0],), dtype = torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['index'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    #yxyx: be warning
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = read_image(image_id, self.train_dir)
        records = self.boxes_df[self.boxes_df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes = xywh2ltrb(boxes)
        return image, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes