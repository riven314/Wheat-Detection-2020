from pathlib import Path
from stuf import stuf


config = stuf(    
    # dir spec
    DATA_PATH = Path('/userhome/34/h3509807/wheat-data'),
    MODEL_DIR = Path('trained_models'),
    PREFIX_NAME = 'final',
    
    # model spec
    ARCH = 'resnet50-coco',
    BIAS = -2,
    GAMMA = 2.,
    ALPHA = 0.25,
    DETECT_THRESHOLD = 0.5,
    NMS_THRESHOLD = 0.3,
    RATIOS = [0.5, 1, 2.],
    SCALES = [1., 0.6, 0.3],
    #SCALES = [1, 2**(-1/3), 2**(-2/3)]
    
    # train loop spec
    TEST_MODE = False,
    RAND_SEED = 144,
    RESIZE_SZ = 256,
    BS = 32,
    IS_FT = False,
    INIT_EPOCH = 10,
    FT_EPOCH = 4,
    INIT_LR = 1e-4,
    FT_LR = slice(1e-6, 5e-4)

)
