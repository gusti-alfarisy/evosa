import tensorflow as tf
import numpy as np

ACTIVATION_FUNC_LIST = {
    0: 'elu',
    1: 'exponential',
    2: 'gelu',
    3: 'hard_sigmoid',
    4: 'linear',
    5: 'relu',
    6: 'selu',
    7: 'sigmoid',
    8: 'softplus',
    9: 'softsign',
    10: 'swish',
    11: 'tanh',
}
# base_model = tf.keras.applications.MobileNetV3Small(
#             input_shape=(224, 224, 3),
#             include_top=False,
#             weights="imagenet"
#         )
# base_model.trainable = True
# base_inputs = base_model.layers[0].input
# base_outputs = base_model.layers[-1].output
# global_pool = tf.keras.layers.GlobalAveragePooling2D()(base_outputs)
# model = tf.keras.Model(inputs=base_inputs, outputs=global_pool)
#
# def get_total_params(model):
#
#     trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
#     # non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
#     non_trainable_params = 0
#     total_params = trainable_params + non_trainable_params
#     # return non_trainable_params
#     return total_params
#
# INDEX_TRAINABLE_LAYER = [i for i, x in enumerate(model.layers) if get_total_params(x) > 0]
# print(INDEX_TRAINABLE_LAYER)

INDEX_TRAINABLE_LAYER = [None, 2, 3, 9, 10, 13, 15, 20, 21, 22, 23, 26, 27, 29, 30, 31, 32, 34, 35, 37, 38, 40, 41, 47, 48, 54, 56, 61, 62, 63, 64, 69, 70, 76, 78, 83, 84, 86, 87, 92, 93, 99, 101, 106, 107, 109, 110, 115, 116, 122, 124, 129, 130, 131, 132, 137, 138, 144, 146, 151, 152, 154, 155, 161, 162, 168, 170, 175, 176, 177, 178, 183, 184, 190, 192, 197, 198, 200, 201, 206, 207, 213, 215, 220, 221, 223, 224]

# TLEncoding = Transfer Learning Encoding
class TLEncoding:

    def __init__(self, encoding=[], max_neuron=2000, optimized_num_layer=3, max_activation=11):

        self.max_neuron = max_neuron
        self.max_activation = max_activation
        self.ol = optimized_num_layer

        self.ub = [self.max_neuron, self.max_neuron, self.max_neuron, 11, 11, 11, 10, 31]

        self.lb = [0, 0, 0, 0, 0, 0, 1, 0]

        if len(encoding) > 0:
            self.encoding = np.array(encoding)
        else:
            self.init_random()
            self.encoding = np.array(self.encoding)

    def init_random(self):
        self.encoding = [np.random.randint(0, self.max_neuron) for i in range(self.ol)]
        for i in range(3):
            self.encoding.append(np.random.randint(0, self.max_activation))

        self.encoding.append(np.random.randint(1, 10))
        self.encoding.append(np.random.randint(0, 32))


    def __rand_float(self, ilub):
        # ilub = index lower and upper bound
        return np.random.uniform(self.lb[ilub], self.ub[ilub], size=(1))[0]

    def num_layers(self):
        return [int(x) for x in self.encoding[:3]]

    def activations(self):
        return [self.get_activation_str(int(round(x, 0))) for x in self.encoding[3:6]]

    def activations_int(self):
        return [int(round(x, 0)) for x in self.encoding[3:6]]

    def patience(self):
        return int(round(self.encoding[6]))

    def get_activation_str(self, index):
        # print("index activation", index)
        return ACTIVATION_FUNC_LIST[index]

    def index_trainable(self):
        return INDEX_TRAINABLE_LAYER[int(round(self.encoding[7]))]

    def __str__(self):
        return f"Encoding: {self.encoding}"

    def optimized_num_layers(self):
        return self.ol

    def is_no_addtional_neurons(self):
        nl = self.num_layers()
        if sum(nl) <= 0:
            return True
        return False

    def key_str(self):
        num_layers = self.num_layers()
        num_layers = [str(x) for x in num_layers]
        activations = self.activations_int()
        activations = [self.get_activation_str(x) for x in activations]
        patience = self.patience()
        itrain = self.index_trainable()

        key_str = f"{' '.join(num_layers)} {' '.join(activations)} {patience} {itrain}"
        return key_str


class MNetGeneratorTrainbaleOne:

    def __init__(self, num_class=10, neuron=[], activation=[], i_trainable=None):
        self.neuron = neuron
        self.num_class = num_class
        self.activation = activation
        self.i_trainable = i_trainable

    def get_model(self):
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=True,
            weights="imagenet"
        )

        base_model.trainable = False

        base_inputs = base_model.layers[0].input
        base_outputs = base_model.layers[-6].output
        temp_dense = base_outputs
        for i, e in enumerate(self.neuron[:3]):
            if e <= 0:
                continue
            temp_dense = tf.keras.layers.Dense(e, activation=self.activation[i])(temp_dense)

        output = tf.keras.layers.Dense(self.num_class)(temp_dense)
        model = tf.keras.Model(inputs=base_inputs, outputs=output)

        # Check this to be important
        # model.layers[238].trainable = True
        if self.i_trainable is not None:
            model.layers[self.i_trainable] = True

        return model

    def get_model_tf(self):
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )

        base_model.trainable = False

        base_inputs = base_model.layers[0].input
        base_outputs = base_model.layers[-1].output
        global_pool = tf.keras.layers.GlobalAveragePooling2D()(base_outputs)
        temp_dense = global_pool
        for i, e in enumerate(self.neuron[:3]):
            if e <= 0:
                continue
            temp_dense = tf.keras.layers.Dense(e, activation=self.activation[i])(temp_dense)

        output = tf.keras.layers.Dense(self.num_class)(temp_dense)
        model = tf.keras.Model(inputs=base_inputs, outputs=output)

        # Check this to be important
        # model.layers[238].trainable = True
        if self.i_trainable is not None:
            model.layers[self.i_trainable] = True

        return model

    def get_model_standard(self):
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        base_inputs = base_model.layers[0].input
        base_outputs = base_model.layers[-1].output
        global_pool = tf.keras.layers.GlobalAveragePooling2D()(base_outputs)
        output = tf.keras.layers.Dense(self.num_class)(global_pool)
        model = tf.keras.Model(inputs=base_inputs, outputs=output)
        return model


if __name__ == "__main__":
    # abc = TopoActivationEncoding()
    # print(abc)
    # print(abc.activations())
    pass
