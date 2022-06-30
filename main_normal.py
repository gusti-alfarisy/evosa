import tensorflow as tf
import numpy as np
import time
from dataset_pool import MALAYA_KEW
from encoder.tapotl import TLEncoding
from objective_functions import cf_TAPOTL_10Fold, fit_kfold_normal, main_cost_kfold_test, \
    objective_res

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_memory_growth(gpus[1], True)

dataset = MALAYA_KEW
dataset.load_dataset()
dataset.load_data_fold_train_test()

dataset.load_dataset()
dataset.load_data_fold_train_test()

def get_predict_data(image_generator):
    for x in image_generator:
        x = x[0][0]
        x = np.reshape(x, [1, 224, 224, 3])
        return x


predict_data = get_predict_data(dataset.val_generator)


def cost_function(x):
    return cf_TAPOTL_10Fold(x, dataset, predict_data)

def average_latency(model, predict_data, n=10, min=0.001, max=0.002):
    def min_max_norm(x, xmin, xmax):
        return (x-xmin)/(xmax-xmin)

    penalty_list = []
    for i in range(n+1):
        t1 = time.time()
        model.predict(predict_data)
        t2 = time.time()
        latency = t2 - t1
        if latency > 10:
            continue

        penalty_list.append(latency)


    penalty_list = np.array(penalty_list)
    mean = np.mean(penalty_list)
    # print(f"penalty list: {penalty_list}")
    # print(f"len penalty list: {len(penalty_list)}")
    return min_max_norm(mean, min, max)


KFOLD = 10
PATIENCE = 5
# 0 36 0 selu sigmoid swish 10 27
# ENCODING = TLEncoding(encoding=[0, 36, 0, 6, 7, 10, 10, 0])
X, Y = dataset.full_data_fold()
mean_acc_train, mean_acc_val, mean_acc_test, total_params, model, mean_top5_train, mean_top5_val, mean_top5_test = fit_kfold_normal(KFOLD, X, Y, dataset, PATIENCE, return_model=True)
# mean_acc_train, mean_acc_val, mean_acc_test, total_params, model, mean_top5_train, mean_top5_val, mean_top5_test = fit_kfold_tf(ENCODING, KFOLD, X, Y, dataset, PATIENCE, return_model=True)
mean_acc_train_01 = mean_acc_train / 100
mean_acc_val_01 = mean_acc_val / 100
mean_acc_test_01 = mean_acc_test / 100
mean_top5_train_01 = mean_top5_train / 100
mean_top5_val_01 = mean_top5_val / 100
mean_top5_test_01 = mean_top5_test / 100
latency = average_latency(model[0], predict_data, n=10, min=0.00001, max=2)
print(f"Latency: {latency}")

total_cost = main_cost_kfold_test(mean_acc_train=mean_acc_train_01,
                                  mean_acc_test=mean_acc_test_01,
                                  total_params=total_params,
                                  latency=latency,
                                  is_no_additional_neurons=True)

res = objective_res(total_cost,
                    train_acc=mean_acc_train_01,
                    val_acc=mean_acc_val_01,
                    test_acc=mean_acc_test_01,
                    total_params=total_params,
                    latency=latency
                    )

res['cost_top5_train'] = 1 - mean_top5_train_01
res['cost_top5_val'] = 1 - mean_top5_val_01
res['cost_top5_test'] = 1 - mean_top5_test_01
res['acc_train'] = mean_acc_train
res['acc_val'] = mean_acc_val
res['acc_test'] = mean_acc_test
res['acc_top5_train'] = mean_top5_train
res['acc_top5_val'] = mean_top5_val
res['acc_top5_test'] = mean_top5_test
print(res)