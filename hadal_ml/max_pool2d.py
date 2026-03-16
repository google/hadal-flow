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


# For details see https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
# and https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks
class MaxPool2D(keras.layers.Layer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="same",  # valid is not supported for encrypted backprop
        is_first_layer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        if strides is None:
            self.strides = [1, 1]
        elif isinstance(strides, int):
            self.strides = [strides, strides]
        else:
            self.strides = strides
        self.padding_str = padding.upper()
        self.is_first_layer = is_first_layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding_str,
                "is_first_layer": self.is_first_layer,
            }
        )
        return config

    def reset_split_forward_mode(self):
        self._layer_intermediate = []
        self._layer_input_shapes = []

    def call(self, inputs, training=False, split_forward_mode=False):
        """Inputs are expected to be in NHWC format, i.e.
        [batch, height, width, channels]
        """

        # Tensorflow 2.18.0 has a bug in the graph optimizer where if the
        # input tensor here is a ReLU op, it will be reordered with the
        # max_pool_with_argmax op below. If this were a normal max_pool
        # op, this would be a legitamate transformation. However, the
        # optimizer does not correctly handle the fact that max_pool_with_argmax
        # returns two values. When it rearranges the ReLU op after MP,
        # it fogets the fact that there are two outputs. This makes it
        # appear as though the `argmax` or `output` tensors are actually
        # the inputs.
        # This TODO in tensorflow may be related:
        # https://github.com/tensorflow/tensorflow/blob/7573c297de8ec216dc85855937b3625f63b0a4e5/tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc#L1734
        #
        # To work around this, we add a 0.0 to the input tensor. Control
        # dependencies and other methods to prevent the optimizer from
        # reordering the ops did not work.
        inputs = inputs + 0.0

        outputs, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            self.pool_size,
            self.strides,
            self.padding_str,
            include_batch_in_index=False,
        )

        if training:
            if split_forward_mode:
                self._layer_intermediate.append(argmax)
                self._layer_input_shapes.append(inputs.shape.as_list())
            else:
                self._layer_intermediate = [argmax]
                self._layer_input_shapes = [inputs.shape.as_list()]

        return outputs

    def backward(self, dy, rotation_key=None, sensitivity_analysis_factor=None):
        """Compute the gradient."""
        if sensitivity_analysis_factor is not None:
            # When performing sensitivity analysis, use the most recent
            # intermediate state.
            indices = self._layer_intermediate[-1]
            input_shape = self._layer_input_shapes[-1]
        else:
            indices = tf.concat(
                [tf.identity(z) for z in self._layer_intermediate], axis=0
            )
            input_shape = self._layer_input_shapes[0]
            input_shape[0] += sum([x[0] for x in self._layer_input_shapes[1:]])
        grad_weights = []

        # On the forward pass, inputs may be batched differently than the
        # ciphertext scheme when not in eager mode. Pad them to match the
        # ciphertext scheme.
        if isinstance(dy, hadal.ShellTensor64):
            padding = [[0, dy._context.num_slots - input_shape[0]]] + [
                [0, 0] for _ in range(len(input_shape) - 1)
            ]
            indices = tf.pad(indices, padding)

        if self.is_first_layer:
            d_x = None  # no gradient needed for first layer
        else:
            d_x = hadal.max_unpool2d(
                dy,
                indices,
                self.pool_size,
                self.strides,
                self.padding_str,
                output_shape=input_shape,
            )

        return grad_weights, d_x

    def unpacking_funcs(self):
        return []
