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


class ShellEmbedding(keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
        skip_embeddings_below_index=0,
        grad_reduction="none",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.skip_embeddings_below_index = skip_embeddings_below_index
        self.grad_reduction = grad_reduction

        if grad_reduction not in ["galois", "none"]:
            raise ValueError(
                f"Invalid grad_reduction type: {grad_reduction} (must be 'galois' or 'none')"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "embeddings_initializer": initializers.serialize(
                    self.embeddings_initializer
                ),
                "skip_embeddings_below_index": self.skip_embeddings_below_index,
                "grad_reduction": self.grad_reduction,
            }
        )
        return config

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.embeddings_initializer,
            dtype=tf.keras.backend.floatx(),
        )

    def reset_split_forward_mode(self):
        self._layer_intermediate = []

    def call(self, inputs, training=False, split_forward_mode=False):
        if inputs.dtype != tf.int64:
            # When using model.fit() keras will cast the input to float.
            inputs = tf.cast(inputs, tf.int64)

        if inputs.ndim != 2:
            raise ValueError(f"Embedding layer expects rank 2 input. Got {inputs}.")

        if training:
            if split_forward_mode:
                self._layer_intermediate.append(inputs)
            else:
                self._layer_intermediate = [inputs]

        outputs = tf.experimental.numpy.take(self.embeddings, inputs, axis=0)
        return outputs

    def backward(self, dy, rotation_key=None, sensitivity_analysis_factor=None):
        """
        dy is shape (batch_size, sentence_length, output_dimension)
        _layer_input is (batch_size, sentence_length)

        This function sums all the dy[a, b, :] where _layer_input[a, b] has the
        same value.

        There are two ways to do the ciphertext aggregation:
          1) rotate -> add -> mask
              First rotate the value selected by the index to slot 0,
              then add the result to a running tally for this index. Once
              everything has been added, mask off everything except the first
              slot, and decrypt.
          2) mask -> add -> reduce sum
              Mask off the value selected by the index with a one-hot vector.
              Add the result to a running tally for this index. Lastly,
              reduce_sum over the batch dimension and decrypt.
        Since rotations are the slowest of the 3 operations, I think it would
        be faster to do option 2, since values in the same slot can be added
        requiring only one rotation.

        The op we need for this layer is a special version of tf.segment_sum()
        and some extra logic to properly handle the batching dimenmsion.
        hadal uses batch axis packing, meaning dy is really shape
        (input_dimension, output_dimension) and each element is a ciphertext
        with (2*batch_size) slots. hadal.segment_sum must pull apart the
        packing dimension of the values by masking with a one-hot.
        """
        if sensitivity_analysis_factor is not None:
            # When performing sensitivity analysis, use the most recent
            # intermediate state.
            indices = self._layer_intermediate[-1]
        else:
            indices = tf.concat(
                [tf.identity(z) for z in self._layer_intermediate], axis=0
            )
        if dy.ndim != indices.ndim + 1:
            raise ValueError(
                f"Embedding layer dy ndims exptected {indices + 1}. Got {dy}."
            )
        # Note: sentence_len = dy.shape[1]

        values = dy

        indices = tf.where(
            indices < self.skip_embeddings_below_index,
            tf.constant(-1, dtype=indices.dtype),
            indices,
        )

        summedvalues, _ = hadal.segment_sum(
            values,
            indices,
            self.input_dim,
            rotation_key,
            reduction=self.grad_reduction,
            skip_pt_counts=True,
        )

        return [summedvalues], None

    @staticmethod
    def unpack(plaintext_packed_dx):
        batch_size = tf.shape(plaintext_packed_dx)[0] // 2
        return plaintext_packed_dx[0, 0] + plaintext_packed_dx[batch_size, 1]

    def unpacking_funcs(self):
        return [ShellEmbedding.unpack]
