import numpy as np
import tensorflow as tf
import hadal_ml
from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Dropout
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import regularizers
from keras.datasets import mnist


def fire_module(input_fire, s1, e1, e3, weight_decay_l2, fireID):
    """
    A wrapper to build fire module

    # Arguments
        input_fire: input activations
        s1: number of filters for squeeze step
        e1: number of filters for 1x1 expansion step
        e3: number of filters for 3x3 expansion step
        weight_decay_l2: weight decay for conv layers
        fireID: ID for the module

    # Return
        Output activations
    """

    # Squezee step
    output_squeeze = Convolution2D(
        s1,
        (1, 1),
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="same",
        name="fire" + str(fireID) + "_squeeze",
        data_format="channels_last",
    )(input_fire)
    # Expansion steps
    output_expand1 = Convolution2D(
        e1,
        (1, 1),
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="same",
        name="fire" + str(fireID) + "_expand1",
        data_format="channels_last",
    )(output_squeeze)
    output_expand2 = Convolution2D(
        e3,
        (3, 3),
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(weight_decay_l2),
        padding="same",
        name="fire" + str(fireID) + "_expand2",
        data_format="channels_last",
    )(output_squeeze)
    # Merge expanded activations
    output_fire = Concatenate(axis=3)([output_expand1, output_expand2])
    return output_fire


def SqueezeNet(num_classes, weight_decay_l2=0.0001, inputs=(128, 128, 3)):
    """
    A wrapper to build SqueezeNet Model

    # Arguments
        num_classes: number of classes defined for classification task
        weight_decay_l2: weight decay for conv layers
        inputs: input image dimensions

    # Return
        A SqueezeNet Keras Model
    """
    input_img = Input(shape=inputs)

    conv1 = Convolution2D(
        32,
        (7, 7),
        activation="relu",
        kernel_initializer="glorot_uniform",
        strides=(2, 2),
        padding="same",
        name="conv1",
        data_format="channels_last",
    )(input_img)

    maxpool1 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool1", data_format="channels_last"
    )(conv1)

    fire2 = fire_module(maxpool1, 8, 16, 16, weight_decay_l2, 2)
    fire3 = fire_module(fire2, 8, 16, 16, weight_decay_l2, 3)
    fire4 = fire_module(fire3, 16, 32, 32, weight_decay_l2, 4)

    maxpool4 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="maxpool4", data_format="channels_last"
    )(fire4)

    fire5 = fire_module(maxpool4, 16, 32, 32, weight_decay_l2, 5)
    # fire6 = fire_module(fire5, 32, 64, 64, weight_decay_l2, 6)
    # fire7 = fire_module(fire6, 32, 64, 64, weight_decay_l2, 7)
    # fire8 = fire_module(fire7, 64, 128, 128, weight_decay_l2, 8)

    # maxpool8 = MaxPooling2D(
    #     pool_size=(2, 2), strides=(2, 2), name="maxpool8", data_format="channels_last"
    # )(fire8)

    # fire9 = fire_module(maxpool8, 64, 128, 128, weight_decay_l2, 9)
    # fire9_dropout = Dropout(0.5, name="fire9_dropout")(fire9)

    conv10 = Convolution2D(
        num_classes,
        (1, 1),
        activation="relu",
        kernel_initializer="glorot_uniform",
        padding="valid",
        name="conv10",
        data_format="channels_last",
    )(
        fire5
    )  # (fire9_dropout) skip last group of fire modules for memory reasons.

    global_avgpool10 = GlobalAveragePooling2D(data_format="channels_last")(conv10)
    softmax = Activation("softmax", name="softmax")(global_avgpool10)
    return input_img, softmax


def MNIST_FF(num_classes, inputs=(28, 28, 1)):
    input_img = Input(shape=inputs)
    x = Flatten()(input_img)
    x = Dense(64, activation="relu")(x)
    # x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)
    return input_img, x


def MNIST_Shell_FF(num_classes, inputs=(28 * 28, 1)):
    input_img = Input(shape=inputs)
    x = hadal_ml.Flatten()(input_img)
    x = hadal_ml.ShellDense(
        64,
        activation=hadal_ml.relu,
        activation_deriv=hadal_ml.relu_deriv,
    )(x)
    # x = hadal_ml.Dropout(0.5)(x)
    x = hadal_ml.ShellDense(
        num_classes,
        activation=tf.nn.softmax,
    )(x)
    return input_img, x


# Simple_CNN models from tensorflow-privacy tutorial. The first conv and maxpool
# layer may be skipped and the model still has ~95% accuracy (plaintext, no
# input image clipping).
def MNIST_Simple_CNN(num_classes, inputs=(28, 28, 1)):
    input_img = Input(shape=inputs)
    x = Convolution2D(
        filters=32,
        kernel_size=4,
        strides=2,
        # padding="same",
        activation="relu",
    )(input_img)
    x = MaxPooling2D(
        pool_size=(2, 2),
        strides=1,
    )(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)
    return input_img, x


def MNIST_Simple_Shell_CNN(num_classes, inputs=(28, 28, 1)):
    input_img = Input(shape=inputs)
    x = hadal_ml.Conv2D(
        filters=32,
        kernel_size=4,
        strides=2,
        activation=hadal_ml.relu,
        activation_deriv=hadal_ml.relu_deriv,
    )(input_img)
    x = hadal_ml.MaxPool2D(
        pool_size=(2, 2),
        strides=1,
    )(x)
    x = hadal_ml.Flatten()(x)
    x = hadal_ml.ShellDense(
        32,
        activation=hadal_ml.relu,
        activation_deriv=hadal_ml.relu_deriv,
    )(x)
    x = hadal_ml.ShellDense(
        num_classes,
        activation=tf.nn.softmax,
    )(x)
    return input_img, x


def MNIST_datasets(crop_by=0, labels_party_dev="CPU:0", features_party_dev="CPU:0"):
    """
    Returns the MNIST datasets with the specified crop size.

    Args:
        crop_by (int): Number of pixels to crop from each side of the images.

    Returns:
        tuple: A tuple containing training and test datasets.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = np.reshape(x_train, (-1, 28, 28, 1)), np.reshape(
        x_test, (-1, 28, 28, 1)
    )
    x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

    if crop_by > 0:
        if crop_by % 2 != 0:
            raise ValueError("crop_by must be an even number, got: {}".format(crop_by))
        x_train = x_train[
            :, crop_by // 2 : 28 - crop_by // 2, crop_by // 2 : 28 - crop_by // 2, :
        ]
        x_test = x_test[
            :, crop_by // 2 : 28 - crop_by // 2, crop_by // 2 : 28 - crop_by // 2, :
        ]

    with tf.device(labels_party_dev):
        labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
        labels_dataset = labels_dataset.batch(2**10)

    with tf.device(features_party_dev):
        features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        features_dataset = features_dataset.batch(2**10)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(32)

    return features_dataset, labels_dataset, val_dataset
