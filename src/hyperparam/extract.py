"""
extract report for each experiment run
"""
import os
import glob
import json

import pandas as pd


cfgs = glob.glob('models/bias*/config.json')

rows = []
for cfg in cfgs:
    report_json = json.load(open(cfg, 'rb'))
    row = dict()
    row['bias'] = report_json['BIAS']
    row['gamma'] = report_json['GAMMA']
    row['alpha'] = report_json['ALPHA']
    row['nms_threshold'] = report_json['NMS_THRESHOLD']
    row['train_loss'], row['valid_loss'], row['valid_mAP'] = report_json['FINAL_RECORD']
    rows.append(row)

df = pd.DataFrame(rows)
