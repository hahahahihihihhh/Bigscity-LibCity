import os.path

import numpy as np
import pandas as pd


eval_prefix = './logs/NYCTAXI20140103/HGCN'
eval_path = os.path.join(eval_prefix, 'avg_result.csv')
# eval_path = './test/SVR_SZ_TAXI_metrics_poly.csv'
# final_eval_file = 'KMHNet_eval.csv'
# st_multihop_s3/
steps = [3, 6, 12]
dct = {}
df = pd.read_csv(eval_path)
metrics = ['masked_MAE', 'masked_RMSE', 'masked_MAPE']
for _metric in metrics:
    for _step in steps:
        if _metric == 'masked_MAPE':
            dct.setdefault(_metric, []).append(str('%.2f' % (np.average(df[_metric][:_step]) * 100)) + "%")
        else:
            dct.setdefault(_metric, []).append('%.2f' % np.average(df[_metric][:_step]))
print(pd.DataFrame(dct, index = ['Horizon@' + str(_step) for _step in steps]))
print(df)