#!/bin/bash
python -m src.hyperparam.search
python -m src.main --MODEL_DIR current_opt_hyperparam --INIT_EPOCH 20 --IS_FT --FT_EPOCH 20
