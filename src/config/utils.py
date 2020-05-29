import os
from stuf import stuf


def update_config(cfg, opt):
    """ update key-value in cfg by key-value in opt """
    for k, v in cfg.items():
        new_v = getattr(opt, k, None)
        if new_v is not None:
	        setattr(cfg, k, new_v)
    return cfg
