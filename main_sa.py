from optimizer.sa_standard import SA
import tensorflow as tf
import numpy as np
import time
from encoder.tapotl import TLEncoding
from objective_functions import cf_TAPOTL_KFold
import os
import argparse

parser = argparse.ArgumentParser(description="Optimizing architecture for transfer learning using EvoSA")
parser.add_argument('--dataset', type=str, default='MalayaKew')
parser.add_argument('--max_trial', type=int, default=1)
parser.add_argument('--iter', type=int, default=150)
parser.add_argument('--k', type=int, default=10)

from utils.run_first import init_directories
init_directories()

args = parser.parse_args()
# ----- Define the dataset here
from dataset_pool import get_dataset
DATASET = get_dataset(args.dataset)
DATASET.load_dataset()
DATASET.load_data_fold_train_test()
# ----- Dataset

# ----- Define the MAX Trial
MAX_TRIAL = args.max_trial
# -----

# ---- Define total iteration
# Default: 150
ITERATION = args.iter
# ----

gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(x, True) for x in gpus]

print("finishing loading dataset")

def get_predict_data(image_generator):
    for x in image_generator:
        x = x[0][0]
        x = np.reshape(x, [1, 224, 224, 3])
        return x


predict_data = get_predict_data(DATASET.val_generator)


def cost_function(x):
    return cf_TAPOTL_KFold(x, DATASET, predict_data, k=args.k)


name_stored = f"RES_SA_{args.dataset}"
machine = os.getenv('COMPUTERNAME')

for i in range(MAX_TRIAL):
    print(f"Start trial {i + 1}")
    start = time.time()
    best = SA(max_time=ITERATION,
              Encoder=TLEncoding,
              Objective=cost_function,
              lb=TLEncoding().lb,
              ub=TLEncoding().ub,
              log_stored_path=f"SA_{name_stored}_{i + 1}.csv",
              best_stored_path=f"SA_{name_stored}_{i + 1}.obj")

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
