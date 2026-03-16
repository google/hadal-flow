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


class TestMaxPoll2d(tf.test.TestCase):
    def _test_max_pool2d_plaintext_forward_backward_correct(
        self, stride, padding, pool_sz, channels
    ):
        """Test that the forward and backward pass of plaintexts with
        hadal_ml.Conv2D has the same output as tf.keras.layers.Conv2D.
        """
        # Create a random input.
        im_shape = [context.num_slots] + [12, 12, channels]
        im = tf.random.uniform(im_shape, minval=0, maxval=5, dtype=tf.int64)
        im = tf.cast(im, tf.float32)

        # Create a shell layer and a keras layer.
        layer = hadal_ml.MaxPool2D(
            pool_size=(pool_sz, pool_sz),
            padding=padding,
            strides=stride,
        )
        tf_layer = keras.layers.MaxPool2D(
            pool_size=(pool_sz, pool_sz),
            padding=padding,
            strides=stride,
        )
        layer.build(im_shape)
        tf_layer.build(im_shape)

        # First check forward pass.
        y = layer(im, training=True)
        tf_y = tf_layer(im)
        self.assertAllClose(y, tf_y)

        # Next check backward pass.
        _, dx = layer.backward(tf.ones_like(y), rotation_key)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(im)
            y = tf_layer(im)
        tf_dx = tape.gradient(y, im)

        self.assertAllClose(dx, tf_dx)

    def test_max_pool2d_plaintext_forward_backward_correct(self):
        # Only test cases where the input_size and (input_size - pool_size) are
        # evenly divisible by the stride. This is due to a limitation in the
        # TensorFlow max_unpool2d function.
        for stride in [1, 2]:
            for padding in ["valid", "same"]:
                for pool_sz in [2, 3, 4]:
                    for channels in [1, 3]:
                        self._test_max_pool2d_plaintext_forward_backward_correct(
                            stride, padding, pool_sz, channels
                        )

    def _test_tf_func(self, stride, padding, pool_sz, channels):
        im_shape = [context.num_slots] + [12, 12, channels]

        layer = hadal_ml.MaxPool2D(
            pool_size=(pool_sz, pool_sz),
            padding=padding,
            strides=stride,
        )
        layer.build(im_shape)

        im = tf.random.uniform(im_shape, minval=0, maxval=3, dtype=tf.int64)
        im = tf.cast(im, tf.float32)

        @tf.function
        def forward_backward(x):
            y = layer(x, training=True)

            dy = tf.ones_like(y)
            enc_dy = hadal.to_encrypted(dy, key, context)

            # Encrypted backward pass.
            _, enc_dx = layer.backward(enc_dy, rotation_key)
            dx = hadal.to_tensorflow(enc_dx, key)

            # Plaintext backward pass.
            _, pt_dx = layer.backward(dy)

            return dx, dx.shape, pt_dx

        dx, dx_shape_inf, pt_dx = forward_backward(im)

        # Check the inferred shapes are the same as the real shapes.
        self.assertAllEqual(dx_shape_inf, dx.shape)

        # Check the values.
        self.assertAllClose(dx, pt_dx)

    def test_tf_func_eager(self):
        tf.config.run_functions_eagerly(True)

        for stride in [1, 2]:
            for padding in ["valid", "same"]:
                for pool_sz in [2, 3, 4]:
                    for channels in [1, 3]:
                        self._test_tf_func(stride, padding, pool_sz, channels)

    def test_tf_func_defer(self):
        tf.config.run_functions_eagerly(False)

        for stride in [1, 2]:
            for padding in ["valid", "same"]:
                for pool_sz in [2, 3, 4]:
                    for channels in [1, 3]:
                        self._test_tf_func(stride, padding, pool_sz, channels)


if __name__ == "__main__":
    unittest.main()
