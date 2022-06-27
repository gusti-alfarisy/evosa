import tensorflow as tf
import numpy as np
import time
from encoder.tapotl import TLEncoding
from objective_functions import cf_TAPOTL_10Fold, cf_TAPOTL_5Fold, cf_TAPOTL_KFold
import os
from optimizer.evo_sa import EVOSA

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
ITERATION = 100
# ----

def get_predict_data(image_generator):
    for x in image_generator:
        x = x[0][0]
        x = np.reshape(x, [1, 224, 224, 3])
        return x

predict_data = get_predict_data(DATASET.val_generator)



def cost_function(x):
    # 5-fold cross validation
    # return cf_TAPOTL_5Fold(x, DATASET, predict_data)
    # k-fold cross validation (k=7)
    # return cf_TAPOTL_KFold(x, DATASET, predict_data, k=7)
    return cf_TAPOTL_10Fold(x, DATASET, predict_data)


gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_memory_growth(gpus[1], True)

name_stored = "RES_EVOSA_MALAYA"
machine = os.getenv('COMPUTERNAME')

for i in range(MAX_TRIAL):
    print(f"Start trial {i+1}")
    start = time.time()
    best = EVOSA(max_time=ITERATION,
                 Encoder=TLEncoding,
                 Objective=cost_function,
                 lb=TLEncoding().lb,
                 ub=TLEncoding().ub,
                 log_stored_path=f"{name_stored}_{i+1}.csv",
                 best_stored_path=f"{name_stored}_{i+1}.obj")

    end = time.time()

    with open(f'output/{name_stored}.txt', 'a') as f:
        f.write(f"-- EXPERIMENT {i + 1}\n")
        f.write(f"-- Machine: {machine}\n\n")
        f.write(f"best solution: {best}\n")
        f.write(f"position: {best.p}\n")
        f.write(f"total runtime: {(end - start) / 60} minutes\n")
        f.write('----------------\n\n')


# start_encoding = [16, 9, 11, 0, 7.36479279, 9.94314109, 5.5544756, 8.31323081]
# start_encoding = [0, 36, 0, 5.50066767, 6.9919435, 9.56952772, 10, 12.46454372]
# best = EVOSA(max_time=100,
#              Encoder=TAPOTLEncoding,
#              Objective=cost_function,
#              lb=TAPOTLEncoding().lb,
#              ub=TAPOTLEncoding().ub,
#              log_stored_path=f"{name_stored}_cont4.csv",
#              best_stored_path=f"{name_stored}_cont4.obj",
#              start_encoding=start_encoding,
#              evo_only=True)
