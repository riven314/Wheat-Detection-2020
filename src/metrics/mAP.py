import torch
import torch.nn as nn

from src.metrics.utils import nms, process_output, cthw2tlbr, decode_bboxs
from src.metrics.mAP_utils import calculate_image_precision


class mAP:
    __name__ = 'mAP'
    name = 'mAP'
    def __init__(self, img_size, ratios, scales, iou_thresholds = None, 
                 detect_threshold = 0.5, nms_threshold = 0.3):
        """
        :param:
            img_size : image size (for rescale image back from [-1, +1])
            iou_thresholds : list of IoU thresholds for taking bbox pred as a match with gt
            detect_threshold : float, a threshold on pred scores (sigmoid)
            nms_threshold : float, threshold for applying Non-max Suppression
        """
        self.img_size = img_size
        self.ratios = ratios
        self.scales = scales
        
        if iou_thresholds is None:
            iou_thresholds = [i for i in map(lambda i: i/100, range(50, 80, 5))]
        self.iou_thresholds = iou_thresholds
        self.detect_threshold = detect_threshold
        self.nms_threshold = nms_threshold
        
    def clean_bboxs_pred(self, preds, i):
        """
        1. change pred bbox from offset format to cthw format
        2. apply NMS to filter out highly overlapping pred bbox
        3. change filtered pred bbox from cthw to tlbr
        4. change from tlbr format to ltrb format (i.e. [x0, y0, x1, y1])
        5. rescale bbox format from [-1, +1] to image size
        
        :param:
            preds : tuple, batch-wise model predictions
            i : index for which batch you wanna take
        """
        cthw_bboxs_pred, scores, clas_pred = process_output(preds, i, 
                                                            ratios = self.ratios, scales = self.scales,
                                                            detect_thresh = self.detect_threshold)
        if scores is None:
            return []
        keep_idxs = nms(cthw_bboxs_pred, scores, self.nms_threshold)
        cthw_bboxs_pred, scores, clas_pred = cthw_bboxs_pred[keep_idxs], scores[keep_idxs], clas_pred[keep_idxs]
        tlbr_bboxs_pred = cthw2tlbr(cthw_bboxs_pred)
        sort_idxs = torch.argsort(scores, dim = -1, descending = True)
        tlbr_bboxs_pred, scores, clas_pred = tlbr_bboxs_pred[sort_idxs], scores[sort_idxs], clas_pred[sort_idxs]
        bboxs_pred = tlbr_bboxs_pred[:, [1, 0, 3, 2]]
        bboxs_pred = bboxs_pred.detach().cpu().numpy()
        bboxs_pred = decode_bboxs(bboxs_pred, img_size = self.img_size)
        return bboxs_pred
        
    def clean_bboxs_gt(self, bboxs_gt, clas_gt):
        """
        1. remove bboxs, class labels with padding
        2. rescale bbox format from [-1, +1] to image size
        
        :param:
            bboxs_gt : tensor, one-sample target bboxes
            clas_gt : tensor, one-sample target class labels
        """
        keep_idxs = torch.nonzero(clas_gt)
        o_bboxs_gt = bboxs_gt[keep_idxs].squeeze().detach().cpu().numpy()
        o_bboxs_gt = decode_bboxs(o_bboxs_gt, self.img_size)
        return o_bboxs_gt
        
        
    def __call__(self, preds, bboxs_gts, clas_gts):
        """
        :param:
            preds : tuple, batch-wise model predictions
            bbox_gts : tensor, batch-wise target bboxes
            clas_gts : tensor, batch-wise target class labels
        """
        mAPs = []
        bn = bboxs_gts.size(0)
        for bi in range(bn):
            bboxs_pred = self.clean_bboxs_pred(preds, bi)
            bboxs_gt, clas_gt = bboxs_gts[bi], clas_gts[bi]
            bboxs_gt = self.clean_bboxs_gt(bboxs_gt, clas_gt)
            
            if len(bboxs_gt) == 0 or len(bboxs_pred) == 0:
                mAP = 1. if len(bboxs_gt) == len(bboxs_pred) else 0.
            else:
                mAP = calculate_image_precision(bboxs_gt, bboxs_pred, 
                                                thresholds = self.iou_thresholds,
                                                form = 'pascal_voc')
            mAPs.append(mAP)
        metrics = sum(mAPs) / len(mAPs)
        return metrics