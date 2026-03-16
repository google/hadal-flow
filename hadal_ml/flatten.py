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
import tensorflow.keras as keras
from tensorflow.python.keras import initializers
import hadal_flow as hadal
from math import prod


class Flatten(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.flat_shape = [prod(input_shape[1:])]

    def reset_split_forward_mode(self):
        pass

    def call(self, inputs, training=False, split_forward_mode=False):
        self.batch_size = tf.shape(inputs)[0]
        return tf.reshape(inputs, [self.batch_size] + self.flat_shape)

    def backward(self, dy, rotation_key=None, sensitivity_analysis_factor=None):
        new_shape = list(self.input_shape)
        # On the forward pass, inputs may be batched differently than the
        # ciphertext scheme when not in eager mode. Pad them to match the
        # ciphertext scheme.
        if isinstance(dy, hadal.ShellTensor64):
            new_shape[0] = tf.identity(dy._context.num_slots)
        else:
            new_shape[0] = dy.shape[0]
        dw = []
        dx = hadal.reshape(dy, new_shape)

        return dw, dx
