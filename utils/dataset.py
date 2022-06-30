import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from tensorflow.python.keras.utils.data_utils import Sequence

ROOT_DATASET = 'dataset/'


class DataSet:

    def __init__(self, folder_name, tot_class, target_size=(224, 224), batch_size=32, root_dir=ROOT_DATASET,
                 tf_dataset=None):
        self.folder_name = folder_name
        self.tot_class = tot_class
        self.dataset_path = f"{root_dir}{folder_name}/"
        self.target_size = target_size
        self.batch_size = batch_size
        self.root_dir = root_dir

    def load_dataset(self, is_search=False, train_name='train', val_name='val', test_name='test'):
        if train_name is not None:
            self.train_dir = os.path.join(self.dataset_path, train_name)

            self.train_generator = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input).flow_from_directory(
                self.train_dir,
                target_size=self.target_size, batch_size=self.batch_size,
                class_mode="binary"
            )

        if val_name is not None:
            self.val_dir = os.path.join(self.dataset_path, val_name)

            self.val_generator = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input).flow_from_directory(
                self.val_dir,
                target_size=self.target_size, batch_size=self.batch_size,
                class_mode="binary"
            )

        if test_name is not None:
            self.test_dir = os.path.join(self.dataset_path, test_name)

            self.test_generator = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input).flow_from_directory(
                self.test_dir,
                target_size=self.target_size, batch_size=self.batch_size,
                class_mode="binary"
            )

        if is_search:
            self.train_search_dir = os.path.join(self.dataset_path, 'train_search')

            self.train_search_generator = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input).flow_from_directory(
                self.train_search_dir,
                target_size=self.target_size, batch_size=self.batch_size,
                class_mode="binary"
            )


    def set_train_image_generator(self, image_generator):
        self.train_generator = image_generator.flow_from_directory(
            self.train_dir,
            target_size=self.target_size, batch_size=self.batch_size,
            class_mode="binary"
        )

    def full_data_train(self):
        all_x = []
        all_y = []
        i = 0
        for x_batch, y_batch in self.train_generator:

            if i < len(self.train_generator):
                all_x.extend(x_batch)
                all_y.extend(y_batch)
            else:
                break
            i += 1

        all_x = np.array(all_x)
        return all_x, all_y

    def full_data_val(self):
        all_x = []
        all_y = []
        i = 0
        for x_batch, y_batch in self.val_generator:

            if i < len(self.val_generator):
                all_x.extend(x_batch)
                all_y.extend(y_batch)
            else:
                break
            i += 1

        all_x = np.array(all_x)
        return all_x, all_y

    def full_data_test(self):
        all_x = []
        all_y = []
        i = 0
        for x_batch, y_batch in self.test_generator:

            if i < len(self.test_generator):
                all_x.extend(x_batch)
                all_y.extend(y_batch)
            else:
                break
            i += 1

        all_x = np.array(all_x)
        return all_x, all_y

    def load_data_fold_train(self):
        X_train, Y_train = self.full_data_train()
        self.X_crossval = X_train
        self.Y_crossval = np.array(Y_train)

    def load_data_fold_train_test(self):
        X_train, Y_train = self.full_data_train()
        X_test, Y_test = self.full_data_test()
        X_CrossVal = np.concatenate((X_train, X_test), axis=0)
        Y_CrossVal = np.concatenate((Y_train, Y_test), axis=0)

        self.X_crossval = X_CrossVal
        self.Y_crossval = Y_CrossVal

    def load_data_fold_train_val(self):
        X_train, Y_train = self.full_data_train()
        X_val, Y_val = self.full_data_val()

        X_CrossVal = np.concatenate((X_train, X_val), axis=0)
        Y_CrossVal = np.concatenate((Y_train, Y_val), axis=0)

        self.X_crossval = X_CrossVal
        self.Y_crossval = Y_CrossVal

    def load_data_fold_train_val_test(self):
        X_train, Y_train = self.full_data_train()
        X_val, Y_val = self.full_data_val()
        X_test, Y_test = self.full_data_test()

        X_CrossVal = np.concatenate((X_train, X_val, X_test), axis=0)
        Y_CrossVal = np.concatenate((Y_train, Y_val, Y_test), axis=0)

        self.X_crossval = X_CrossVal
        self.Y_crossval = Y_CrossVal

    def full_data_fold(self):
        return self.X_crossval, self.Y_crossval

    def add_train_image_generator(self, image_generator):
        augmented_generator = image_generator.flow_from_directory(
            self.train_dir,
            target_size=self.target_size, batch_size=self.batch_size,
            class_mode="binary"
        )
        # combined_generator = itertools.chain(self.train_generator, augmented_generator)
        # self.train_generator = MergedGenerators(self.train_generator, augmented_generator)
        return AugmentedDataSet(self.train_generator, augmented_generator)


class AugmentedDataSet(Sequence):
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2
        self.len_gen = len(self.generator1)

    def __len__(self):
        return len(self.generator1) + len(self.generator2)

    def __getitem__(self, index):
        if index < self.len_gen:
            return self.generator1[index]
        else:
            return self.generator2[index - self.len_gen]
