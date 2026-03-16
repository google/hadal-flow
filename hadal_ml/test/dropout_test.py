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

context = hadal.create_context64(
    log_n=12,
    main_moduli=[288230376151760897, 288230376152137729],
    plaintext_modulus=4294991873,
    scaling_factor=3,
    seed="test_seed",
)

key = hadal.create_key64(context)


class TestDropout(tf.test.TestCase):
    def _test_dropout_forward(self, per_batch):
        # First check plaintext forward pass.
        x = tf.random.uniform((context.num_slots, 100)) + 1

        dropout_layer = hadal_ml.ShellDropout(0.2, per_batch=per_batch)

        notrain_y = dropout_layer(x, training=False)
        self.assertAllEqual(notrain_y, x)

        train_y = dropout_layer(x, training=True)
        self.assertLess(
            tf.math.count_nonzero(train_y), tf.size(train_y, out_type=tf.int64)
        )

        enc_x = hadal.to_encrypted(x, key, context)
        dropout_layer = hadal_ml.ShellDropout(0.2, per_batch=per_batch)

        notrain_enc_y = dropout_layer.call(enc_x, training=False)
        self.assertAllClose(
            hadal.to_tensorflow(notrain_enc_y, key),
            x,
            atol=1 / context.scaling_factor,
        )

        enc_train_y = dropout_layer.call(enc_x, training=True)
        dec_train_y = hadal.to_tensorflow(enc_train_y, key)
        self.assertLess(
            tf.math.count_nonzero(dec_train_y), tf.size(dec_train_y, out_type=tf.int64)
        )

    def _test_dropout_back(self, per_batch):
        x = tf.random.uniform((context.num_slots, 100)) + 1

        dropout_layer = hadal_ml.ShellDropout(0.2, per_batch=per_batch)

        notrain_y = dropout_layer.call(x, training=True)
        dy = tf.ones_like(notrain_y)

        dw, dx = dropout_layer.backward(dy)

        enc_dy = hadal.to_encrypted(dy, key, context)
        enc_dw, enc_dx = dropout_layer.backward(enc_dy)
        dec_dx = hadal.to_tensorflow(enc_dx, key)

        self.assertEmpty(dw)
        self.assertEmpty(enc_dw)

        self.assertAllClose(dx, dec_dx, atol=1 / context.scaling_factor)

    def test_dropout(self):
        self._test_dropout_forward(False)
        self._test_dropout_back(False)

    def test_dropout_per_batch(self):
        self._test_dropout_forward(True)
        self._test_dropout_back(True)


if __name__ == "__main__":
    unittest.main()
