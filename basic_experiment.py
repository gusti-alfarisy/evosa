import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from dataset_pool import UBD_BOTANICAL_SMALL, UBD_BOTANICAL, VNPLANTS, FOOD101, DEBUG_10FOLD, FOOD101_OSR
from encoder.tapotl import TLEncoding
from my_utils.pretrained_model import MobileNetV3Small, MobileNetV3Large, MobileNetV3Large_Softmax
from objective_functions import cf_TAPOTL_10Fold

dataset = FOOD101_OSR
# dataset.load_dataset()
dataset.load_dataset(val_name="unknown")
# dataset.load_data_fold_train_test()

def get_total_params(model):
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    return total_params


def create_model():
    # model = MobileNetV3Large_Softmax(dataset.tot_class)
    model = MobileNetV3Large(dataset.tot_class)
    return model

import os
checkpoint_filepath = os.path.join('dataset', 'cpt', 'food101_osr', 'food101_softmax')
model = create_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# model.load_weights(checkpoint_filepath)
# res = model.evaluate(dataset.test_generator, batch_size=32)
# print(res)
print(model.summary())



#


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath+"-{epoch:02d}-{val_accuracy:.2f}",
    save_weights_only=True,
    monitor='val_accuracy',
    save_freq='epoch'
)

res = model.fit_generator(dataset.train_generator, epochs=10, validation_data=dataset.test_generator, callbacks=[model_checkpoint_callback])
print(res.history)
print(res.history['accuracy'])
print(res.history['val_accuracy'])

plt.plot(np.arange(1, 11), res.history['accuracy'], label="train accuracy")
plt.plot(np.arange(1, 11), res.history['val_accuracy'], label='test accuracy')
plt.legend(loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, 11))
plt.savefig('plt_acc_food101_osr_nonsoftmax_dell.png')
