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
from hadal_ml.activation import (
    serialize_activation,
    deserialize_activation,
)


# For details see https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
# and https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks
class Conv2D(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding_str="VALID",
        activation=None,
        activation_deriv=None,
        kernel_initializer="glorot_uniform",
        is_first_layer=False,
        grad_reduction="none",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = [1, int(strides), int(strides), 1]
        if padding_str.upper() not in ["SAME", "VALID"]:
            raise ValueError(
                f"Invalid padding_str: {padding_str} (must be 'SAME' or 'VALID')"
            )
        self.padding_str = padding_str.upper()
        self.activation = deserialize_activation(activation)
        self.activation_deriv = deserialize_activation(activation_deriv)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.is_first_layer = is_first_layer
        self.grad_reduction = grad_reduction

        if grad_reduction not in ["galois", "fast", "none"]:
            raise ValueError(
                f"Invalid grad_reduction type: {grad_reduction} (must be 'galois', 'fast', or 'none')"
            )

    def reset_split_forward_mode(self):
        self._layer_intermediate = []
        self._layer_input = []
        self._layer_input_shape = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides[1],
                "padding_str": self.padding_str,
                "activation": serialize_activation(self.activation),
                "activation_deriv": serialize_activation(self.activation_deriv),
                "kernel_initializer": self.kernel_initializer,
                "is_first_layer": self.is_first_layer,
                "grad_reduction": self.grad_reduction,
            }
        )
        return config

    def build(self, input_shape):
        self.in_channels = int(input_shape[-1])

        # Calculate padding now that the input shape is known.
        if self.padding_str == "SAME":
            input_height = input_shape[1]
            input_width = input_shape[2]

            # Calculate output dimensions
            out_height = (input_height + self.strides[1] - 1) // self.strides[1]
            out_width = (input_width + self.strides[2] - 1) // self.strides[2]

            # Calculate total padding needed
            pad_height = max(
                0, (out_height - 1) * self.strides[1] + self.kernel_size - input_height
            )
            pad_width = max(
                0, (out_width - 1) * self.strides[2] + self.kernel_size - input_width
            )
            pad_top = pad_height // 2
            pad_left = pad_width // 2
            self.padding = [
                pad_top,
                pad_height - pad_top,
                pad_left,
                pad_width - pad_left,
            ]
        elif self.padding_str == "VALID":
            self.padding = [0, 0, 0, 0]
        self.tf_padding = [
            [0, 0],
            [self.padding[0], self.padding[1]],  # top, bottom
            [self.padding[2], self.padding[3]],  # left, right
            [0, 0],
        ]

        self.add_weight(
            shape=[self.kernel_size, self.kernel_size, self.in_channels, self.filters],
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=tf.keras.backend.floatx(),
            name="kernel",
        )

    def call(self, inputs, training=False, split_forward_mode=False):
        """Inputs are expected to be in NHWC format, i.e.
        [batch, height, width, channels]
        """
        kernel = tf.identity(self.weights[0])

        outputs = tf.nn.conv2d(
            inputs, kernel, strides=self.strides, padding=self.tf_padding
        )

        if training:
            if split_forward_mode:
                if self._layer_input_shape is None:
                    self._layer_input_shape = inputs.shape.as_list()
                else:
                    self._layer_input_shape[0] += inputs.shape.as_list()[0]
                self._layer_intermediate.append(outputs)
                self._layer_input.append(inputs)
            else:
                self._layer_intermediate = [outputs]
                self._layer_input = [inputs]
                self._layer_input_shape = inputs.shape.as_list()

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def backward(self, dy, rotation_key=None, sensitivity_analysis_factor=None):
        """Compute the gradient."""
        kernel = tf.identity(self.weights[0])

        if sensitivity_analysis_factor is not None:
            # When performing sensitivity analysis, use the most recent
            # intermediate state.
            x = self._layer_input[-1]
            z = self._layer_intermediate[-1]

            # To perform sensitivity analysis, assume the worst case rounding for
            # the intermediate state, dictated by the sensitivity_analysis_factor.
            x = hadal.largest_case_rounding(x, sensitivity_analysis_factor)
            z = hadal.largest_case_rounding(z, sensitivity_analysis_factor)
            kernel = hadal.largest_case_rounding(kernel, sensitivity_analysis_factor)
        else:
            x = tf.concat([tf.identity(x) for x in self._layer_input], axis=0)
            z = tf.concat([tf.identity(z) for z in self._layer_intermediate], axis=0)

        grad_weights = []
        batch_size = tf.shape(x)[0]

        # On the forward pass, x may be batched differently than the
        # ciphertext scheme. Pad them to match the ciphertext scheme.
        if isinstance(dy, hadal.ShellTensor64):
            with tf.name_scope("conv_pad"):
                batch_padding = [
                    [0, dy._context.num_slots - self._layer_input_shape[0]]
                ] + [[0, 0] for _ in range(len(self._layer_input_shape) - 1)]
                x = tf.pad(x, batch_padding)

        if self.activation_deriv is not None:
            dy = self.activation_deriv(z, dy)

        if self.is_first_layer:
            d_x = None  # no gradient needed for first layer
        else:
            exp_kernel = tf.expand_dims(kernel, axis=0)
            exp_kernel = tf.repeat(exp_kernel, batch_size, axis=0)
            d_x = hadal.conv2d_transpose(
                dy,
                exp_kernel,
                strides=self.strides,
                padding=self.padding,
                output_shape=x.shape.as_list(),
            )

        # Swap strides and dilations for the backward pass.
        dy_exp = hadal.expand_dims(dy, axis=3)
        d_w = hadal.conv2d(
            x,
            dy_exp,
            strides=[1, 1, 1, 1],
            padding=self.padding,
            dilations=self.strides,
            with_channel=True,
            output_shape=[
                -1,
                self.kernel_size,
                self.kernel_size,
                self.in_channels,
                self.filters,
            ],
        )

        if self.grad_reduction == "galois":
            d_w = hadal.reduce_sum(d_w, 0, rotation_key)
        elif self.grad_reduction == "fast":
            d_w = hadal.fast_reduce_sum(d_w)

        grad_weights.append(d_w)

        return grad_weights, d_x

    @staticmethod
    def unpack(plaintext_packed_dx):
        batch_size = tf.shape(plaintext_packed_dx)[0] // 2
        return plaintext_packed_dx[0] + plaintext_packed_dx[batch_size]

    def unpacking_funcs(self):
        return [Conv2D.unpack]
