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


class GlobalAveragePooling1D(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset_split_forward_mode(self):
        self._layer_intermediate = 0

    def call(self, inputs, training=False, split_forward_mode=False):
        if training:
            self._layer_intermediate = tf.shape(tf.identity(inputs))[1]

        outputs = tf.reduce_sum(inputs, axis=1)
        outputs /= tf.cast(tf.shape(inputs)[1], tf.keras.backend.floatx())

        return outputs

    def backward(self, dy, rotation_key=None, sensitivity_analysis_factor=None):
        avg_dim = tf.identity(self._layer_intermediate)
        dx = hadal.expand_dims(dy, axis=1)
        dx = hadal.broadcast_to(dx, (hadal.shape(dx)[0], avg_dim, hadal.shape(dx)[2]))

        return [], dx

    @staticmethod
    def unpack(packed_dw):
        return packed_dw

    def unpacking_funcs(self):
        return [GlobalAveragePooling1D.unpack]
