from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import tensorflow as tf
import numpy as np
from encoder.tapotl import TLEncoding
from objective_functions import cf_TAPOTL_10Fold, restart_memoize
import os
import time

from optimizer.sa_standard import Solution2

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

# dataset.load_dataset(train_name="train_cross", val_name="val_earlystop", test_name=None)
# dataset.load_data_fold_train()

def get_predict_data(image_generator):
    for x in image_generator:
        x = x[0][0]
        x = np.reshape(x, [1, 224, 224, 3])
        return x


predict_data = get_predict_data(DATASET.test_generator)
# predict_data = get_predict_data(dataset.val_generator)


def cost_function(x):
    return cf_TAPOTL_10Fold(x, DATASET, predict_data)



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)


# self.ub = [self.max_neuron, self.max_neuron, self.max_neuron, 11, 11, 11, 10, 31]
#
#         self.lb = [0, 0, 0, 0, 0, 0, 1, 0]

COUNTER = 0
def black_box_function(x1, x2, x3, x4, x5, x6, x7, x8):
    """Function with unknown internals we wish to maximize.

    """

    return 1/cost_function([x1, x2, x3, x4, x5, x6, x7, x8])['cost']

# pbounds
encoder = TLEncoding()

pbounds = {
    'x1': (0, encoder.max_neuron), 'x2': (0, encoder.max_neuron),
    'x3': (0,encoder.max_neuron), 'x4': (0, encoder.max_activation),
    'x5': (0,encoder.max_activation), 'x6':(0,encoder.max_activation),
    'x7': (1,10), 'x8': (0, 31)
}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
    # verbose=2
)

name_stored = "RES_BO_MALAYA_FIXX"
machine = os.getenv('COMPUTERNAME')


for i in range(MAX_TRIAL):
    restart_memoize()
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
        # verbose=2
    )
    logger = JSONLogger(path=f"output/bo/{name_stored}_{i+1}.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    start = time.time()
    optimizer.maximize(init_points=2, n_iter=ITERATION)
    end = time.time()


    print("MAX results", optimizer.max)
    print(f"real cost: {1/optimizer.max['target']}")
    p_optimized = list(optimizer.max['params'].values())
    encoder = TLEncoding(encoding=p_optimized)
    res = cost_function(encoder.encoding)

    best = Solution2(encoder.encoding,
                              res['cost'],
                              res['cost_train'],
                              res['cost_val'],
                              res['cost_test'],
                              res['cost_top5_train'],
                              res['cost_top5_val'],
                              res['cost_top5_test'],
                              encoder.key_str(),
                              res['total_params'],
                              res['latency'])

    print(f"FINAL SOLUTION {i+1}-----")
    print(best)
    print(f"total runtime: {(end - start) / 60} minutes\n")
    with open(f'output/{name_stored}.txt', 'a') as f:
        f.write(f"-- EXPERIMENT {i + 1}\n")
        f.write(f"-- Machine: {machine}\n\n")
        f.write(f"best solution: {best}\n")
        f.write(f"total runtime: {(end - start) / 60} minutes\n")
        f.write('----------------\n\n')