import tensorflow as tf


def MobileNetV3Small(tot_class):
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=True,
        weights="imagenet"
    )
    base_model.trainable = False

    base_inputs = base_model.layers[0].input
    base_outputs = base_model.layers[-6].output
    output = tf.keras.layers.Dense(tot_class)(base_outputs)
    model = tf.keras.Model(inputs=base_inputs, outputs=output)
    return model

def MobileNetV3Large(tot_class):
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    base_inputs = base_model.layers[0].input
    base_outputs = base_model.layers[-1].output
    x = tf.keras.layers.GlobalAvgPool2D()(base_outputs)
    x = tf.keras.layers.Dense(tot_class)(x)
    model = tf.keras.Model(inputs=base_inputs, outputs=x)
    return model

def MobileNetV3Large_Softmax(tot_class):
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    base_inputs = base_model.layers[0].input
    base_outputs = base_model.layers[-1].output
    x = tf.keras.layers.GlobalAvgPool2D()(base_outputs)
    x = tf.keras.layers.Dense(tot_class)(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=base_inputs, outputs=x)
    return model

# model = MobileNetV3Large_Softmax(101)
# print(model.summary())
