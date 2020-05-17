"""
credit:
- Competition metric details + script: https://www.kaggle.com/pestipeti/competition-metric-details-script?scriptVersionId=33780809
"""
import os
from pdb import set_trace

import torch
import numpy as np
import numba
from numba import jit, prange

from typing import List, Union, Tuple


#from functools import partial
#thresholds = [i for i in map(lambda i: i/100, range(50, 80, 5))]
#mAP_getter = partial(calculate_image_precision, thresholds = thresholds)


def mAP(b_preds, b_bboxs_gts, b_clas_gts, 
        thresholds: Union[List, Tuple], 
        detection_thre = 0.5,
        form: str = 'coco') -> float:
    """ ** assume b_preds are filtered by detection_threshold! """
    b_clas_preds, b_bboxs_preds, sizes = b_preds
    b_iter = zip(b_clas_preds, b_bboxs_preds, b_clas_gts, b_bboxs_gts)
    
    b_mets = []
    for clas_preds, bbox_preds, clas_gts, bbox_gts in b_iter:
        clas_preds = clas_preds.cpu().numpy().squeeze()
        bbox_preds = bbox_preds.cpu().numpy()
        clas_gts = clas_gts.cpu().numpy()
        bbox_gts = bbox_gts.cpu().numpy()
        
        set_trace()
        # filter out trivial ground truth
        
        
        # filter out preds below detection threshold and sort by confidence
        preds_idxs = np.argwhere(clas_preds >= detection_thre).squeeze()
        clas_preds = clas_preds[preds_idxs]
        bbox_preds = bbox_preds[preds_idxs]
        #sort_idxs = 
        
        # restore predicted bbox [x, y, w, h]
        
        
        met = calculate_image_precision(gts, preds, thresholds, form)
        b_mets.append(met)
        
    return sum(b_mets) / len(b_mets)


@jit(nopython=True)
def calculate_iou(gt: List[Union[int, float]], 
                  pr: List[Union[int, float]], 
                  form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (List[Union[int, float]]) coordinates of the ground-truth box
        pr: (List[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0])
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1])

    if (dx <= 0) or (dy <= 0):
        return 0.0
    else:
        overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0]) * (gt[3] - gt[1]) +
            (pr[2] - pr[0]) * (pr[3] - pr[1]) -
            overlap_area
    )

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(gts: List[List[Union[int, float]]],
                    pred: List[Union[int, float]],
                    threshold: float = 0.5,
                    form: str = 'pascal_voc') -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    # for gt_idx, ggt in enumerate(gts):
    for gt_idx in range(len(gts)):
        iou = calculate_iou(gts[gt_idx], pred, form=form)

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def np_delete_workaround(arr: np.ndarray, idx: int):
    """Deletes element by index from a ndarray.

    Numba does not handle np.delete, so this workaround
    needed for the fast MAP calculation.

    Args:
        arr: (np.ndarray) numpy array
        idx: (int) index of the element to remove

    Returns:
        (np.ndarray) New array
    """
    mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
    mask[idx] = False

    return arr[mask]


@jit(nopython=True)
def calculate_precision(gts: List[List[Union[int, float]]],
                        preds: List[List[Union[int, float]]],
                        threshold: float = 0.5,
                        form: str = 'coco') -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], threshold=threshold, form=form)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts = np_delete_workaround(gts, best_match_gt_idx)

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = len(gts)

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts: List[List[Union[int, float]]],
                              preds: List[List[Union[int, float]]],
                              thresholds: Union[List, Tuple] = (0.5, ),
                              form: str = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts, preds, threshold=threshold, form=form)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


#thresholds = numba.typed.List()
#for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
#    thresholds.append(x)
