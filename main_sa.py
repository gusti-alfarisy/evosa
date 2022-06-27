import optimizer.sa_standard as meta
import tensorflow as tf
import numpy as np
import time
from encoder.tapotl import TLEncoding
from objective_functions import cf_TAPOTL_10Fold
import os

# ----- Define the dataset here
from dataset_pool import MALAYA_KEW
DATASET = MALAYA_KEW
DATASET.load_dataset()
DATASET.load_data_fold_train_test()
# ----- Dataset

# ----- Define the MAX Trial
MAX_TRIAL = 1
# -----

# ---- Define total iteration
ITERATION = 150
# ----

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_memory_growth(gpus[1], True)

print("finishing loading dataset")

def get_predict_data(image_generator):
    for x in image_generator:
        x = x[0][0]
        x = np.reshape(x, [1, 224, 224, 3])
        return x


predict_data = get_predict_data(DATASET.test_generator)
# predict_data = get_predict_data(dataset.val_generator)


def cost_function(x):
    return cf_TAPOTL_10Fold(x, DATASET, predict_data)
    # return cf_TAPOTL_5Fold(x, dataset, predict_data)


name_stored = "RES_SA_MALAYA_FIXX"
machine = os.getenv('COMPUTERNAME')

for i in range(MAX_TRIAL):
    print(f"Start trial {i + 1}")
    start = time.time()
    best = meta.SA(max_time=ITERATION,
                   Encoder=TLEncoding,
                   Objective=cost_function,
                   lb=TLEncoding().lb,
                   ub=TLEncoding().ub,
                   log_stored_path=f"{name_stored}_{i+1}.csv",
                   best_stored_path=f"{name_stored}_{i+1}.obj")

    end = time.time()
    # res = train_TAPL(dataset, TAPOTLEncoding(encoding=best.p))
    # best_model = res['model']
    # best_solution = Solution(best.p, cost=best.cost, cost_train=res['cost_train'], cost_val=['cost_val'],
    #                          total_params=best.total_params, latency=0)
    # best_model.save(f'dataset/models/cso/{name_stored}_{i + 1}.h5')

    with open(f'output/{name_stored}.txt', 'a') as f:
        f.write(f"-- GPUs - EXPERIMENT 150 ITER: {i + 1}\n")
        f.write(f"-- Machine: {machine}\n\n")
        f.write(f"best solution: {best}\n")
        f.write(f"Original Position: {best.p}\n")
        f.write(f"total runtime: {(end - start) / 60} minutes\n")
        f.write('----------------\n\n')
