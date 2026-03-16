# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
import numpy as np
import hadal_flow as hadal


def relu(x):
    return tf.nn.relu(x)


def relu_deriv(y, dy):
    assert not isinstance(y, hadal.ShellTensor64)
    # Cannot operate on individual slots of a shell tensor.
    # Formulate the problem as element-wise multiplication.
    mask = tf.where(
        y > 0,
        tf.constant(1, dtype=tf.keras.backend.floatx()),
        tf.constant(0, dtype=tf.keras.backend.floatx()),
    )

    # Use a special hadal function to multiply by a binary mask, without
    # affecting the scaling factor in the case when dy is a ShellTensor.
    return hadal.mask_with_pt(dy, mask)


def sigmoid(x):
    return tf.math.sigmoid(x)


def sigmoid_deriv(y, d_y):
    return d_y * (y * (1 - y))


def serialize_activation(activation):
    if activation is None:
        return None
    elif isinstance(activation, str):
        return activation
    elif callable(activation):
        return activation.__name__
    raise ValueError(f"Invalid activation: {activation}")


def deserialize_activation(activation):
    if activation is None:
        return None
    elif isinstance(activation, str):
        return globals()[activation]
    elif callable(activation):
        return activation
    raise ValueError(f"Invalid activation: {activation}")
