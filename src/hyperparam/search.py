"""
brute force run experiments to search optimal hyperparameters
"""
import os

from src.main import train_run
from src.config.retinanet import config
from src.config.utils import update_config


if __name__ == '__main__':
    for bias in [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0]:
        for gamma in [1]:
            for alpha in [0.5]:
                for nms in [0.3]:
                    config.BIAS = bias
                    config.GAMMA = gamma
                    config.ALPHA = alpha
                    config.NMS_THRESHOLD = nms
                    config.MODEL_DIR = f'bias{bias}_gamma{gamma}_alpha{alpha}_nms{nms}'
                    print(f'training: {config.MODEL_DIR}')
                    train_run(config)
