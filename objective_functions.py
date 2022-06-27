import copy

import numpy as np
import time

from sklearn.model_selection import KFold

# from encoder.ta_patience_trainable import TAPTEncoding
from encoder.tapotl import TLEncoding, MNetGeneratorTrainbaleOne
# from encoder.tapt_selective_layer import TAPSLEncoding
# from encoder.topoactiv_epoch import TopoActivEpochEncoding
# from encoder.topoactiv_optim import TopoActivOptimEncoding, OptimizerGenerator
# from encoder.topoactive_epoch_trainable import TopoActivEpochTrainableEncoding, MNetGeneratorTrainbale
# from encoder.topologies_activation import MNetGenerator, TopoActivationEncoding
import tensorflow as tf

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
EVALUATION_LIST = {}


def log_base(val, base):
    return np.log(val) / np.log(base)


def restart_memoize():
    global EVALUATION_LIST
    EVALUATION_LIST = {}




# in seconds
def average_latency(model, predict_data, n=10, min=0.001, max=0.002):
    def min_max_norm(x, xmin, xmax):
        return (x-xmin)/(xmax-xmin)

    penalty_list = []
    for i in range(n+1):
        t1 = time.time()
        model.predict(predict_data)
        t2 = time.time()
        duration = t2 - t1
        # print(f"Duration {i}: {duration}")
        if i == 0:
            continue
        penalty_list.append(duration)

    penalty_list = np.array(penalty_list)
    mean = np.mean(penalty_list)
    # print(f"mean penalty list: {mean}")
    return min_max_norm(mean, min, max)

def fit_kfold_from_model(model=None, k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)
    total_params = get_total_params(model)

    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def fit_kfold(encoder=None, k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)

    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                      neuron=encoder.num_layers(),
                                      activation=encoder.activations(),
                                      i_trainable=encoder.index_trainable()
                                      ).get_model()

    total_params = get_total_params(model)

    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()
        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                          neuron=encoder.num_layers(),
                                          activation=encoder.activations(),
                                          i_trainable=encoder.index_trainable()
                                          ).get_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def fit_kfold_tf(encoder=None, k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)

    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                      neuron=encoder.num_layers(),
                                      activation=encoder.activations(),
                                      i_trainable=encoder.index_trainable()
                                      ).get_model_tf()

    total_params = get_total_params(model)

    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()
        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                          neuron=encoder.num_layers(),
                                          activation=encoder.activations(),
                                          i_trainable=encoder.index_trainable()
                                          ).get_model_tf()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def fit_kfold_normal(k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)

    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                      ).get_model_tf()

    total_params = get_total_params(model)

    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()
        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                          ).get_model_tf()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def fit_kfold_normal_tf(k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)

    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                      ).get_model_standard()

    total_params = get_total_params(model)
    # print(model.summary())
    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()
        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                          ).get_model_standard()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def fit_kfold_normal_1024(k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)

    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class, neuron=[1024], activation=['relu']
                                      ).get_model_tf()
    # print(model.summary())
    total_params = get_total_params(model)

    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()

        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class, neuron=[1024], activation=['relu']
                                          ).get_model_tf()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def fit_kfold_normal_1536(k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)

    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class, neuron=[1536], activation=['relu']
                                      ).get_model_tf()

    total_params = get_total_params(model)

    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()

        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class, neuron=[1536], activation=['relu']
                                          ).get_model_tf()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def fit_kfold_normal_512(k=10, X=None, Y=None, dataset=None, patience=3, return_model=False):
    kfold = KFold(n_splits=k, shuffle=True)

    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class, neuron=[512], activation=['relu']
                                      ).get_model_tf()

    total_params = get_total_params(model)

    i_fold = 1
    acc_test_per_fold = []
    acc_train_per_fold = []
    acc_val_per_fold = []
    loss_per_fold = []
    acc_top5_train_per_fold = []
    acc_top5_val_per_fold = []
    acc_top5_test_per_fold = []

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    if return_model:
        model_list = []

    for train, test in kfold.split(X, Y):
        # model = create_model()

        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class, neuron=[512], activation=['relu']
                                          ).get_model_tf()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        res = model.fit(X[train], Y[train], batch_size=32, epochs=100, verbose=0, validation_data=dataset.val_generator,
                        callbacks=[callback])

        # print("In objectives")
        # print(res.history)
        score = model.evaluate(X[test], Y[test], verbose=0)
        acc_test_per_fold.append(score[1] * 100)
        acc_train_per_fold.append(res.history['accuracy'][-1] * 100)
        acc_val_per_fold.append(res.history['val_accuracy'][-1] * 100)
        acc_top5_train_per_fold.append(res.history['sparse_top_k_categorical_accuracy'][-1] * 100)
        acc_top5_val_per_fold.append(res.history['val_sparse_top_k_categorical_accuracy'][-1] * 100)
        loss_per_fold.append(score[0])
        acc_top5_test_per_fold.append(score[2] * 100)
        # print("FOLD KE", i_fold)

        i_fold += 1

        if return_model:
            model_list.append(model)

    mean_acc_train_fold = np.mean(np.array(acc_train_per_fold))
    mean_acc_val_fold = np.mean(np.array(acc_val_per_fold))
    mean_acc_test_fold = np.mean(np.array(acc_test_per_fold))
    mean_top5_train = np.mean(np.array(acc_top5_train_per_fold))
    mean_top5_val = np.mean(np.array(acc_top5_val_per_fold))
    mean_top5_test = np.mean(np.array(acc_top5_test_per_fold))
    if return_model:
        return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, model_list, mean_top5_train, mean_top5_val, mean_top5_test

    return mean_acc_train_fold, mean_acc_val_fold, mean_acc_test_fold, total_params, mean_top5_train, mean_top5_val, mean_top5_test

def memoize_eval_topo_activation(F):
    evaluation_list = {}

    def inner(x):
        encoding = TopoActivationEncoding(encoding=x)
        key_str = encoding.key_str()
        if key_str not in evaluation_list:
            evaluation_list[key_str] = F(x)
        else:
            print("memoize is used", key_str)

        return evaluation_list[key_str]

    return inner


def results(res_model):
    return {
        'accuracy': res_model.history['accuracy'][-1],
        'val_accuracy': res_model.history['val_accuracy'][-1],
        'val_loss': res_model.history['val_loss'][-1],
        'loss': res_model.history['loss'][-1]
    }


def callback_es(patience):
    return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)


def objective_res(total_cost, train_acc, val_acc, test_acc, total_params, latency):
    return {
        'cost': total_cost,
        'cost_train': 1 - train_acc,
        'cost_val': 1 - val_acc,
        'cost_test': 1 - test_acc,
        'total_params': total_params,
        'latency': latency
    }


def cf_TAPOTL(p, dataset, predict_data=None):
    encoder = TLEncoding(encoding=p)
    key_str = encoder.key_str()

    if key_str not in EVALUATION_LIST:
        # print("evaluate even with memoization", key_str)
        model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class,
                                          neuron=encoder.num_layers(),
                                          activation=encoder.activations(),
                                          i_trainable=encoder.index_trainable()
                                          ).get_model()

        max_epoch = 100
        patience = encoder.patience()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # data_generator = dataset.train_search_generator if is_search else dataset.train_generator
        callback_cost = callback_es(patience)

        res = model.fit_generator(dataset.train_generator, epochs=max_epoch, validation_data=dataset.val_generator,
                                  callbacks=[callback_cost], verbose=0)
        res = results(res)
        accuracy_train = res['accuracy']
        accuracy_val = res['val_accuracy']
        # val_loss = res.history['val_loss'][-1]

        # t1 = time.time()
        # model.predict(predict_data)
        # latency = time.time() - t1
        latency = 0

        is_no_additional_neurons = encoder.is_no_addtional_neurons()
        print("is_no_additional_neurons", is_no_additional_neurons)
        total_params = get_total_params(model)
        total_cost = main_cost_function(accuracy_train, accuracy_val, total_params, latency, is_no_additional_neurons)

        res = objective_res(total_cost,
                            train_acc=accuracy_train,
                            val_acc=accuracy_val,
                            total_params=total_params,
                            latency=latency
                            )

        EVALUATION_LIST[key_str] = res
        return res
    else:
        print("memoize is used", key_str)

    return EVALUATION_LIST[key_str]


def train_TAPOTL(dataset, p):
    encoder = TLEncoding(encoding=p)
    model = MNetGeneratorTrainbaleOne(num_class=dataset.tot_class, neuron=encoder.num_layers(),
                                      activation=encoder.activations()).get_model()

    max_epoch = 100
    patience = encoder.patience()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    callback_func = callback_es(patience)
    res = model.fit_generator(dataset.train_generator, epochs=max_epoch, validation_data=dataset.val_generator,
                              callbacks=[callback_func], verbose=0)

    accuracy_train = res.history['accuracy'][-1]
    accuracy_val = res.history['val_accuracy'][-1]
    return {
        'accuracy_train': accuracy_train * 100,
        'accuracy_val': accuracy_val * 100,
        'cost_train': 1 - accuracy_train,
        'cost_val': 1 - accuracy_val,
        'model': model
    }


def cf_TAPOTL_KFold(p, dataset, predict_data=None, k=10):
    encoder = TLEncoding(encoding=p)
    key_str = encoder.key_str()

    if key_str not in EVALUATION_LIST:
        # print("evaluate even with memoization", key_str)

        patience = encoder.patience()

        X, Y = dataset.full_data_fold()
        mean_acc_train, mean_acc_val, mean_acc_test, total_params, model, mean_top5_train, mean_top5_val, mean_top5_test = fit_kfold_tf(encoder, k, X, Y, dataset, patience, return_model=True)
        mean_acc_train_01 = mean_acc_train / 100
        mean_acc_val_01 = mean_acc_val / 100
        mean_acc_test_01 = mean_acc_test / 100
        mean_top5_train_01 = mean_top5_train / 100
        mean_top5_val_01 = mean_top5_val / 100
        mean_top5_test_01 = mean_top5_test / 100
        latency = average_latency(model[0], predict_data, n=10, min=0.00001, max=2)

        is_no_additional_neurons = encoder.is_no_addtional_neurons()
        # print("is_no_additional_neurons", is_no_additional_neurons)
        # total_cost = main_cost_function(mean_acc_train_01, accuracy_val, total_params, latency, is_no_additional_neurons)
        # total_cost = main_cost_function_kfold(mean_acc=mean_acc_train_01,
        # total_cost = main_cost_function_kfold(mean_acc=mean_acc_train_01,
        #                                       total_params=total_params,
        #                                       latency=latency,
        #                                       is_no_additional_neurons=is_no_additional_neurons)

        total_cost = main_cost_kfold_test(mean_acc_train=mean_acc_train_01,
                                          mean_acc_test=mean_acc_test_01,
                                          total_params=total_params,
                                          latency=latency,
                                          is_no_additional_neurons=is_no_additional_neurons)

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
        EVALUATION_LIST[key_str] = res
        return res
    else:
        print("memoize is used", key_str)

    return EVALUATION_LIST[key_str]


# TAPOTL with K-Fold cross validation
def cf_TAPOTL_10Fold(p, dataset, predict_data=None):
    return cf_TAPOTL_KFold(p, dataset, predict_data=predict_data, k=10)


def cf_TAPOTL_5Fold(p, dataset, predict_data=None):
    return cf_TAPOTL_KFold(p, dataset, predict_data, k=5)

def train_TAPOTL_KFOLD_withModel(dataset, p, k=10):
    encoder = TLEncoding(encoding=p)
    patience = encoder.patience()

    X, Y = dataset.full_data_fold()
    mean_acc_train, mean_acc_val, mean_acc_test, total_params, model_list = fit_kfold(encoder, k, X, Y, dataset, patience, return_model=True)
    mean_acc_train_01 = mean_acc_train / 100
    mean_acc_val_01 = mean_acc_val / 100
    mean_acc_test_01 = mean_acc_test / 100
    latency = 0

    is_no_additional_neurons = encoder.is_no_addtional_neurons()

    total_cost = main_cost_kfold_test(mean_acc_train=mean_acc_train_01,
                                      mean_acc_test=mean_acc_test_01,
                                      total_params=total_params,
                                      latency=latency,
                                      is_no_additional_neurons=is_no_additional_neurons)

    res = objective_res(total_cost,
                        train_acc=mean_acc_train_01,
                        val_acc=mean_acc_val_01,
                        test_acc=mean_acc_test_01,
                        total_params=total_params,
                        latency=latency
                        )

    return res, model_list

def train_TAPOTL_KFOLD(dataset, p, k=10):
    encoder = TLEncoding(encoding=p)
    patience = encoder.patience()

    X, Y = dataset.full_data_fold()
    mean_acc_train, mean_acc_val, mean_acc_test, total_params = fit_kfold(encoder, k, X, Y, dataset, patience)
    mean_acc_train_01 = mean_acc_train / 100
    mean_acc_val_01 = mean_acc_val / 100
    mean_acc_test_01 = mean_acc_test / 100
    latency = 0

    is_no_additional_neurons = encoder.is_no_addtional_neurons()
    # print("is_no_additional_neurons", is_no_additional_neurons)
    # total_cost = main_cost_function(mean_acc_train_01, accuracy_val, total_params, latency, is_no_additional_neurons)
    # total_cost = main_cost_function_kfold(mean_acc=mean_acc_train_01,
    # total_cost = main_cost_function_kfold(mean_acc=mean_acc_train_01,
    #                                       total_params=total_params,
    #                                       latency=latency,
    #                                       is_no_additional_neurons=is_no_additional_neurons)

    total_cost = main_cost_kfold_test(mean_acc_train=mean_acc_train_01,
                                      mean_acc_test=mean_acc_test_01,
                                      total_params=total_params,
                                      latency=latency,
                                      is_no_additional_neurons=is_no_additional_neurons)

    res = objective_res(total_cost,
                        train_acc=mean_acc_train_01,
                        val_acc=mean_acc_val_01,
                        test_acc=mean_acc_test_01,
                        total_params=total_params,
                        latency=latency
                        )

    return res

# def train_topoactivation(dataset, encoder):
#     model = MNetGenerator(num_class=dataset.tot_class, neuron=encoder.num_layers(),
#                           activation=encoder.activations()).get_model()
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy']
#     )
#
#     res = model.fit_generator(dataset.train_generator, epochs=100, validation_data=dataset.val_generator,
#                               callbacks=[callback], verbose=0)
#     accuracy_train = res.history['accuracy'][-1]
#     accuracy_val = res.history['val_accuracy'][-1]
#     return {
#         'accuracy_train': accuracy_train * 100,
#         'accuracy_val': accuracy_val * 100,
#         'cost_train': 1 - accuracy_train,
#         'cost_val': 1 - accuracy_val,
#         'model': model
#     }


# def train_topoactivation_epoch(dataset, encoder: TopoActivEpochEncoding):
#     model = MNetGenerator(num_class=dataset.tot_class, neuron=encoder.num_layers(),
#                           activation=encoder.activations()).get_model()
#     max_epoch = encoder.max_epoch()
#     patience = encoder.patience()
#     is_early_stop = encoder.is_using_early_stop()
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy']
#     )
#     if is_early_stop:
#         callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
#         res = model.fit_generator(dataset.train_generator, epochs=max_epoch, validation_data=dataset.val_generator,
#                                   callbacks=[callback], verbose=0)
#     else:
#         res = model.fit_generator(dataset.train_generator, epochs=max_epoch, validation_data=dataset.val_generator,
#                                   verbose=0)
#
#     accuracy_train = res.history['accuracy'][-1]
#     accuracy_val = res.history['val_accuracy'][-1]
#     return {
#         'accuracy_train': accuracy_train * 100,
#         'accuracy_val': accuracy_val * 100,
#         'cost_train': 1 - accuracy_train,
#         'cost_val': 1 - accuracy_val,
#         'model': model
#     }


# def train_topoactivation_epoch_trainable(dataset, encoder: TopoActivEpochTrainableEncoding):
#     model = MNetGeneratorTrainbale(num_class=dataset.tot_class, neuron=encoder.num_layers(),
#                                    activation=encoder.activations()).get_model()
#
#     max_epoch = encoder.max_epoch()
#     patience = encoder.patience()
#     is_early_stop = encoder.is_using_early_stop()
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=['accuracy']
#     )
#
#     if is_early_stop:
#         callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
#         res = model.fit_generator(dataset.train_generator, epochs=max_epoch, validation_data=dataset.val_generator,
#                                   callbacks=[callback], verbose=0)
#     else:
#         res = model.fit_generator(dataset.train_generator, epochs=max_epoch, validation_data=dataset.val_generator,
#                                   verbose=0)
#
#     accuracy_train = res.history['accuracy'][-1]
#     accuracy_val = res.history['val_accuracy'][-1]
#     return {
#         'accuracy_train': accuracy_train * 100,
#         'accuracy_val': accuracy_val * 100,
#         'cost_train': 1 - accuracy_train,
#         'cost_val': 1 - accuracy_val,
#         'model': model
#     }


def get_total_params(model):
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    return total_params


def main_cost_function_kfold(mean_acc, total_params, latency=1, tradeoff=0.95, min_params=1_500_000,
                             is_no_additional_neurons=False):
    # avg_accuracy = (accuracy + accuracy_val) / 2
    # avg_accuracy = mean_acc
    # print("avg_accuracy", avg_accuracy)
    # if avg_accuracy > tradeoff:
    param_penalty = (total_params - min_params) / (min_params * 2)
    # else:
    #     param_penalty = ((total_params - min_params) / (min_params * 2)) + 0.1

    # latency_penalty = latency / 100
    latency_penalty = 0

    penalty_no_neuron = 0

    if is_no_additional_neurons:
        penalty_no_neuron = 1

    cost = (1 - mean_acc) + param_penalty + latency_penalty + penalty_no_neuron
    return cost


def main_cost_kfold_test(mean_acc_train, mean_acc_test, total_params, latency=1, tradeoff=0.95, min_params=1_500_000,
                         is_no_additional_neurons=False):
    # avg_accuracy = (accuracy + accuracy_val) / 2
    # avg_accuracy = mean_acc
    # print("avg_accuracy", avg_accuracy)
    # if avg_accuracy > tradeoff:
    # param_penalty = (total_params - min_params) / (min_params * 3)
    param_penalty = total_params/1_000_000 - 1.5
    # param_penalty = log_base(total_params / 1_000_000, 15)
    # else:
    #     param_penalty = ((total_params - min_params) / (min_params * 2)) + 0.1

    # latency_penalty = latency / 100
    latency_penalty = latency

    no_neuron_penalty = 0

    if is_no_additional_neurons:
        no_neuron_penalty = 1

    # cost = (1 - mean_acc_train) + (1 - mean_acc_test) + param_penalty + latency_penalty + penalty_no_neuron
    cost = 0.5 * (1 - mean_acc_test) + 0.25 * param_penalty + 0.25 * latency_penalty + no_neuron_penalty
    # cost = 0.9 * (1 - mean_acc_test) + 0.05 * param_penalty + 0.05 * latency_penalty + no_neuron_penalty
    return cost


def main_cost_function(accuracy, accuracy_val, total_params, latency=1, tradeoff=0.95, min_params=1_500_000,
                       no_additional_neuron=False):
    avg_accuracy = (accuracy + accuracy_val) / 2
    # print("avg_accuracy", avg_accuracy)
    # if avg_accuracy > tradeoff:
    param_penalty = (total_params - min_params) / (min_params * 2)
    # else:
    #     param_penalty = ((total_params - min_params) / (min_params * 2)) + 0.1

    # latency_penalty = latency / 100
    latency_penalty = 0

    penalty_no_neuron = 0

    if no_additional_neuron:
        penalty_no_neuron = 1

    if accuracy > accuracy_val and accuracy >= 1:
        cost = (1 - accuracy) + (1 - accuracy_val) + np.abs(
            1 - np.exp((accuracy - accuracy_val) + 0.2)) + param_penalty + latency_penalty + penalty_no_neuron
    else:
        cost = (1 - accuracy) + (1 - accuracy_val) + param_penalty + latency_penalty + penalty_no_neuron

    # print("Penalty")
    # print(
    #     f"normal cost: {1 - accuracy} | penalty overfit: {np.abs(1 - np.exp((accuracy - accuracy_val) + 0.2))} | param penalty: {param_penalty} | latency penalty: {latency_penalty}")
    return cost


def main_cost_function_loss(accuracy, accuracy_val, loss_val, total_params, latency=1, tradeoff=0.95,
                            min_params=1_500_000):
    avg_accuracy = (accuracy + accuracy_val) / 2
    # print("avg_accuracy", avg_accuracy)
    # if avg_accuracy > tradeoff:
    param_penalty = (total_params - min_params) / (min_params * 2)
    # else:
    #     param_penalty = ((total_params - min_params) / (min_params * 2)) + 0.1

    # latency_penalty = latency / 100
    latency_penalty = 0

    print("loss val", loss_val)
    if accuracy > accuracy_val and accuracy >= 1:
        cost = (1 - accuracy) + (1 - accuracy_val) + loss_val + np.abs(
            1 - np.exp((accuracy - accuracy_val) + 0.2)) + param_penalty + latency_penalty
    else:
        cost = (1 - accuracy) + (1 - accuracy_val) + loss_val + param_penalty + latency_penalty

    # print("Penalty")
    # print(
    #     f"normal cost: {1 - accuracy} | penalty overfit: {np.abs(1 - np.exp((accuracy - accuracy_val) + 0.2))} | param penalty: {param_penalty} | latency penalty: {latency_penalty}")
    return cost




def cost_function_topo_activation(p, dataset, predict_data=None, is_search=False):
    encoder = TopoActivationEncoding(encoding=p)
    key_str = encoder.key_str()

    # print("Key dictionary")
    # print(evaluation_list.keys())
    # print("Key string to be chekced", key_str)
    if key_str not in EVALUATION_LIST:
        # print("evaluate even with memoization", key_str)
        model = MNetGenerator(num_class=dataset.tot_class, neuron=encoder.num_layers(),
                              activation=encoder.activations()).get_model()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        data_generator = dataset.train_search_generator if is_search else dataset.train_generator

        res = model.fit_generator(data_generator, epochs=100, validation_data=dataset.val_generator,
                                  callbacks=[callback], verbose=0)

        accuracy_train = res.history['accuracy'][-1]
        accuracy_val = res.history['val_accuracy'][-1]

        # t1 = time.time()
        # model.predict(predict_data)
        # latency = time.time() - t1
        latency = 0

        total_params = get_total_params(model)
        total_cost = main_cost_function(accuracy_train, accuracy_val, total_params, latency)
        # res = {
        #     'cost': total_cost,
        #     'cost_train': 1 - accuracy_train,
        #     'cost_val': 1 - accuracy_val,
        #     'total_params': total_params,
        #     'latency': latency,
        #     'model': model
        # }
        res = {
            'cost': total_cost,
            'cost_train': 1 - accuracy_train,
            'cost_val': 1 - accuracy_val,
            'total_params': total_params,
            'latency': latency
            # 'model': None
        }

        EVALUATION_LIST[key_str] = res
        return res
    else:
        print("memoize is used", key_str)

    return EVALUATION_LIST[key_str]



def penalty_overfitting(train_loss, val_loss):
    # print(val_loss - train_loss)
    if train_loss < 0.1 and val_loss < 0.1:
        print("ada yang masuk loss < 0.1")
        return 0

    return np.exp(val_loss - train_loss)
    # pass


# print(penalty_overfitting(0.015, 1.02))

def main_cf_loss(train_loss, val_loss, train_accuracy, val_accuracy, total_params, latency=1, threshold=0.95,
                 min_params=1_500_000):
    avg_accuracy = (train_accuracy + val_accuracy) / 2
    # print("avg_accuracy", avg_accuracy)
    if avg_accuracy > threshold:
        param_penalty = (total_params - min_params + 1e-5) / (min_params * 4)
    else:
        param_penalty = ((total_params - min_params + 1e-5) / (min_params * 2))

    # latency_penalty = latency / 100
    latency_penalty = 0
    of_penalty = penalty_overfitting(train_loss, val_loss)
    cost = train_loss + val_loss + of_penalty + param_penalty + latency_penalty
    # cost = train_loss + val_loss + param_penalty + latency_penalty

    # if train_accuracy > val_accuracy:
    #     cost = (1 - accuracy) + np.abs(1 - np.exp((accuracy - accuracy_val) + 0.2)) + param_penalty + latency_penalty
    # else:
    #     cost = (1 - accuracy) + param_penalty + latency_penalty

    # print("Penalty")
    # print(
    #     f"normal cost: {1 - accuracy} | penalty overfit: {np.abs(1 - np.exp((accuracy - accuracy_val) + 0.2))} | param penalty: {param_penalty} | latency penalty: {latency_penalty}")
    # print("of penalty", of_penalty)
    # print("total cost", cost)
    return cost