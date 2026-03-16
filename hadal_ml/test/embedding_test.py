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
import unittest
import tensorflow as tf
import keras
import numpy as np
import hadal_flow as hadal
import hadal_ml

# Num plaintext bits: 32, noise bits: 84
# Max representable value: 654624
context = hadal.create_context64(
    log_n=9,
    main_moduli=[288230376151748609, 144115188076060673],
    plaintext_modulus=4294991873,
    scaling_factor=3,
    seed="test_seed",
)

key = hadal.create_key64(context)
rotation_key = hadal.create_rotation_key64(context, key)


class TestEmbedding(tf.test.TestCase):
    def test_embedding_forward(self):
        input_dim = 100
        output_dim = 10
        embedding_layer = hadal_ml.ShellEmbedding(input_dim, output_dim)

        # First check plaintext forward pass.
        x = tf.zeros((context.num_slots, 5), dtype=tf.float32)
        y = embedding_layer(x)

        # Check that the output is the same for the same inputs.
        for i in range(1, context.num_slots):
            self.assertAllEqual(y[0], y[i])

        # Next check encrypted forward pass throws an error.
        enc_x = hadal.to_encrypted(x, key, context)
        try:
            should_fail = embedding_layer(enc_x)
        except:
            pass
        else:
            raise ValueError(
                "Embedding layer forward with encrypted value should fail."
            )

    def _test_embedding(self, reduction):
        input_dim = 100
        output_dim = 16
        embedding_layer = hadal_ml.ShellEmbedding(
            input_dim,
            output_dim,
            skip_embeddings_below_index=1,
            grad_reduction=reduction,
        )

        sentence_length = 3
        special_index = 2
        x = (
            tf.ones((context.num_slots, sentence_length), dtype=tf.float32)
            * special_index
        )

        @tf.function
        def forward_backward(x):
            y = embedding_layer(x, training=True)

            dy = tf.ones_like(y)
            enc_dy = hadal.to_encrypted(dy, key, context)

            enc_dw, _ = embedding_layer.backward(enc_dy, rotation_key)
            dw = hadal.to_tensorflow(enc_dw[0], key)

            # Plaintext backward pass.
            pt_dws, _ = embedding_layer.backward(dy)
            pt_dw = pt_dws[0]

            if reduction == "none":
                dw = tf.reduce_sum(dw, axis=0)
                pt_dw = tf.reduce_sum(pt_dw, axis=0)
            else:
                dw = embedding_layer.unpack(dw)
                pt_dw = embedding_layer.unpack(pt_dw)

            return dw, enc_dw[0].shape, pt_dw, pt_dws[0].shape

        # dw, shape_inf = forward_backward(x)
        dw, dw_shape_inf, pt_dw, pt_dw_shape_inf = forward_backward(x)

        # Check the inferred shape of the gradient is correct.
        if reduction == "none":
            self.assertAllEqual(
                dw_shape_inf, [context.num_slots, input_dim, output_dim]
            )
            self.assertAllEqual(
                pt_dw_shape_inf, [context.num_slots, input_dim, output_dim]
            )
        else:
            self.assertAllEqual(
                dw_shape_inf, [context.num_slots, 2, input_dim, output_dim]
            )
            self.assertAllEqual(
                pt_dw_shape_inf, [context.num_slots, 2, input_dim, output_dim]
            )

        for i in range(0, input_dim):
            # Check dw[special_index] has counted the number of elements.
            if i == special_index:
                self.assertAllEqual(
                    dw[special_index, :],
                    tf.constant(
                        context.num_slots * sentence_length, shape=(output_dim,)
                    ),
                )
                self.assertAllEqual(
                    pt_dw[special_index, :],
                    tf.constant(
                        context.num_slots * sentence_length, shape=(output_dim,)
                    ),
                )
            # Make sure the rest of the gradient elements are 0.
            else:
                self.assertAllEqual(dw[i, :], tf.constant(0, shape=(output_dim,)))
                self.assertAllEqual(pt_dw[i, :], tf.constant(0, shape=(output_dim,)))

    def test_embedding_eager(self):
        tf.config.run_functions_eagerly(True)
        self._test_embedding("galois")
        self._test_embedding("none")

    def test_embedding_defer(self):
        tf.config.run_functions_eagerly(False)
        self._test_embedding("galois")
        self._test_embedding("none")


if __name__ == "__main__":
    unittest.main()
