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


class TestConv2D(tf.test.TestCase):
    def _test_conv2d_plaintext_forward_backward_correct(
        self, im_sz, kern_sz, stride, padding, filters, channels
    ):
        """Test that the forward and backward pass of plaintexts with
        hadal_ml.Conv2D has the same output as tf.keras.layers.Conv2D.
        """
        # Create a random input.
        im_shape = [context.num_slots] + [im_sz, im_sz, channels]
        im = tf.random.uniform(im_shape, minval=0, maxval=5, dtype=tf.int64)
        im = tf.cast(im, tf.float32)

        # Create a shell layer and a keras layer with the same integer-valued
        # weights.
        conv_layer = hadal_ml.Conv2D(
            filters=filters,
            kernel_size=kern_sz,
            padding_str=padding,
            strides=stride,
        )
        tf_conv_layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kern_sz,
            padding=padding,
            strides=stride,
            use_bias=False,
        )
        conv_layer.build(im_shape)
        tf_conv_layer.build(im_shape)
        rand_weights = tf.random.uniform(
            [kern_sz, kern_sz, channels, filters], minval=0, maxval=2, dtype=tf.int64
        )
        rand_weights = tf.cast(rand_weights, tf.float32)
        conv_layer.set_weights([rand_weights])
        tf_conv_layer.set_weights([rand_weights])

        # First check forward pass.
        y = conv_layer(im, training=True)
        tf_y = tf_conv_layer(im)
        self.assertAllClose(y, tf_y)

        # Next check backward pass.
        dws, dx = conv_layer.backward(tf.ones_like(y), rotation_key)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(im)
            y = tf_conv_layer(im)
        tf_dx = tape.gradient(y, im)
        tf_dw = tape.gradient(y, tf_conv_layer.trainable_variables)

        self.assertAllClose(dx, tf_dx)
        self.assertAllClose(tf.reduce_sum(dws[0], axis=0), tf_dw[0])

    def test_conv2d_plaintext_forward_backward_correct(self):
        for im_kern_sz in [(10, 5), (11, 4), (28, 8)]:
            for stride in [1, 2]:
                for padding in ["valid", "same"]:
                    for filters in [3, 4]:
                        for channels in [1, 3]:
                            self._test_conv2d_plaintext_forward_backward_correct(
                                im_kern_sz[0],
                                im_kern_sz[1],
                                stride,
                                padding,
                                filters,
                                channels,
                            )

    def _test_tf_func(self, im_sz, kern_sz, stride, padding, filters, channels):
        conv_layer = hadal_ml.Conv2D(
            filters=filters,
            kernel_size=kern_sz,
            strides=stride,
            padding_str=padding,
        )

        im_shape = [context.num_slots] + [im_sz, im_sz, channels]
        im = tf.random.uniform(im_shape, minval=0, maxval=5, dtype=tf.int64)
        im = tf.cast(im, tf.float32)

        # Assign integer weights to the layer for easy comparison between
        # plaintext and HE-based backprop.
        conv_layer.build(im_shape)
        rand_weights = tf.random.uniform(
            [kern_sz, kern_sz, channels, filters], minval=0, maxval=2, dtype=tf.int64
        )
        rand_weights = tf.cast(rand_weights, tf.float32)
        conv_layer.set_weights([rand_weights])

        @tf.function
        def forward_backward(x):
            y = conv_layer(x, training=True)

            dy = tf.ones_like(y)
            enc_dy = hadal.to_encrypted(dy, key, context)

            # Encrypted backward pass.
            enc_dw, enc_dx = conv_layer.backward(enc_dy, rotation_key)
            dw = hadal.to_tensorflow(enc_dw[0], key)
            # dw = conv_layer.unpack(dw)  # for layer reduction 'fast' or 'galois'
            dx = hadal.to_tensorflow(enc_dx, key)

            # Plaintext backward pass.
            pt_dws, pt_dx = conv_layer.backward(dy, None)
            pt_dw = pt_dws[0]  # No unpack required for pt.

            return dw, dx, dw.shape, dx.shape, pt_dw, pt_dx

        dw, dx, dw_shape_inf, dx_shape_inf, pt_dw, pt_dx = forward_backward(im)

        # Check the inferred shapes are the same as the real shapes.
        self.assertAllEqual(dw_shape_inf, tf.shape(dw))
        self.assertAllEqual(dx_shape_inf, tf.shape(dx))

        # Check the values.
        self.assertAllClose(dw, pt_dw)
        self.assertAllClose(dx, pt_dx)

    def _test_tf_func_configs(self):
        for im_kern_sz in [(10, 5), (11, 4), (28, 8)]:
            for stride in [1, 2]:
                for padding in ["valid", "same"]:
                    for filters in [3, 4]:
                        for channels in [1, 3]:
                            self._test_tf_func(
                                im_kern_sz[0],
                                im_kern_sz[1],
                                stride,
                                padding,
                                filters,
                                channels,
                            )

    def test_tf_func_eager(self):
        tf.config.run_functions_eagerly(True)
        self._test_tf_func_configs()

    def test_tf_func_defer(self):
        tf.config.run_functions_eagerly(False)
        self._test_tf_func_configs()


if __name__ == "__main__":
    unittest.main()
