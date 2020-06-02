import cv2
import numpy as np

from src.common.utils import read_image, xywh2ltrb


def grid_idxs(idx, n_col = 4):
    """ get row, col index for plt.subplots axs """
    row = idx // n_col
    col = idx % n_col
    return row, col


def bboxs_on_image(image, bboxs):
    """ 
    overlay bboxs on an image
    
    :param:
        img : np.array (H, W, C)
        bboxs : list of [x0, y0, x1, y1]
    :return:
        img_wbboxs : np.array (H, W, C)
    """
    cp_image = image.copy()
    for x0, y0, x1, y1 in bboxs:
        cv2.rectangle(cp_image, (x0, y0), (x1, y1), (0, 1, 0), 2)
    return cp_image


def plot_source_samples(src, sample_n = 8, figsize = (16, 16)):
    """ 
    src: 'usask_1', 'arvalis_1', 'inrae_1', 'ethz_1', 'arvalis_3', 'rres_1', 'arvalis_2' 
    """
    tgt_df = df_folds[df_folds.source == src]
    image_ids = tgt_df.sample(sample_n).index.values
    fig, axs = plt.subplots(2, 4, figsize = figsize)

    for idx, image_id in enumerate(image_ids):
        image = read_image(image_id)
        bboxs = np.array(marking[marking.image_id == image_id][['x', 'y', 'w', 'h']])
        bboxs = xywh2ltrb(bboxs)
        image = bboxs_on_image(image, bboxs)
        row, col = grid_idxs(idx, col_n = int(sample_n // 2))
        axs[row, col].imshow(image)
    plt.tight_layout();
    
    return image_ids
