#!/bin/bash
python -i -m src.hyperparam.search
python -i -m src.main --MODEL_DIR current_opt_hyperparam --INIT_EPOCH 20 --IS_FT --FT_EPOCH 20
