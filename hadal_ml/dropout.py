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


class ShellDropout(keras.layers.Layer):
    def __init__(
        self,
        rate,
        noise_shape=None,
        seed=None,
        per_batch=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rate = float(rate)
        if (self.rate < 0.0) or (self.rate >= 1.0):
            raise ValueError(
                "rate must be a float in the range [0, 1), got {}".format(self.rate)
            )
        self.noise_shape = noise_shape
        self.seed = seed
        self.per_batch = per_batch
        if self.per_batch and self.noise_shape is not None:
            raise ValueError("noise_shape must be None when per_batch is True")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "noise_shape": self.noise_shape,
                "seed": self.seed,
                "per_batch": self.per_batch,
            }
        )
        return config

    def build(self, input_shape):
        self.units_in = int(input_shape[1])

    def reset_split_forward_mode(self):
        self._layer_intermediate = []

    def call(self, inputs, training=False, split_forward_mode=False):
        if not training or self.rate == 0.0:
            return inputs

        if self.per_batch:
            dummy_input = tf.ones([1] + inputs.shape[1:])
        else:
            dummy_input = tf.ones(inputs.shape)

        dropout_mask = tf.nn.dropout(
            dummy_input,
            self.rate,
            noise_shape=self.noise_shape,
            seed=self.seed,
        )

        if training:
            if split_forward_mode:
                self._layer_intermediate.append(dropout_mask)
            else:
                self._layer_intermediate = [dropout_mask]

        output = inputs * dropout_mask
        return output

    def backward(self, dy, rotation_key=None, sensitivity_analysis_factor=None):
        if sensitivity_analysis_factor is not None:
            # When performing sensitivity analysis, use the most recent
            # intermediate state.
            dropout_mask = self._layer_intermediate[-1]

            # To perform sensitivity analysis, assume the worst case rounding
            # for the intermediate state, dictated by the
            # sensitivity_analysis_factor.
            # Note: The dropout mask is not necessarily 0 or 1.
            dropout_mask = hadal.largest_case_rounding(
                dropout_mask, sensitivity_analysis_factor
            )
        else:
            dropout_mask = tf.concat(
                [tf.identity(z) for z in self._layer_intermediate], axis=0
            )

        d_x = dy * dropout_mask
        return [], d_x
