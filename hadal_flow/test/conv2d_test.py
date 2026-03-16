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
import hadal_flow as hadal
import test_utils
import math


class TestConv2d(tf.test.TestCase):
    test_context = None

    @classmethod
    def setUpClass(cls):
        cls.test_context = test_utils.TestContext(
            outer_shape=[],
            plaintext_dtype=tf.int32,
            log_n=11,
            main_moduli=[8556589057, 8388812801],
            aux_moduli=[],
            plaintext_modulus=40961,
            scaling_factor=1,
        )

    @classmethod
    def tearDownClass(cls):
        cls.rotation_test_contexts = None

    def _test_conv2d_pt_ct(
        self,
        test_context,
        im_shape,
        filter_shape,
        stride,
        padding,
        dilations,
        with_channel=False,
        transpose=False,
    ):
        im_shape = [test_context.shell_context.num_slots] + im_shape
        filter_shape = [test_context.shell_context.num_slots] + filter_shape

        # im = tf.random.uniform(im_shape, minval=0, maxval=3, dtype=tf.int64)
        # im = tf.cast(im, tf.float32)
        im = tf.ones(im_shape, dtype=tf.float32)
        filt = tf.random.uniform(filter_shape, minval=0, maxval=10, dtype=tf.int64)
        filt = tf.cast(filt, tf.float32)
        # filt = tf.ones(filter_shape, dtype=tf.float32)

        e_im = hadal.to_encrypted(im, test_context.key, test_context.shell_context)
        p_filt = hadal.to_shell_plaintext(filt, test_context.shell_context)

        if transpose:
            e_out = hadal.conv2d_transpose(e_im, p_filt, stride, padding, with_channel)
        else:
            e_out = hadal.conv2d(e_im, p_filt, stride, padding, dilations, with_channel)
        out = hadal.to_tensorflow(e_out, test_context.key)

        # Check against the plaintext implementation.
        if transpose:
            check = hadal.conv2d_transpose(im, filt, stride, padding, with_channel)
        else:
            check = hadal.conv2d(im, filt, stride, padding, dilations, with_channel)

        self.assertAllClose(out, check, atol=1e-3)

        # Make sure the arguments were not modified.
        self.assertAllClose(hadal.to_tensorflow(e_im, test_context.key), im)
        self.assertAllClose(hadal.to_tensorflow(p_filt, test_context.key), filt)

    def test_conv2d_pt_ct(self):
        test_configs = [
            # fmt: off
            # Context,          im_shape,    filter_shape, stride,       padding,      dilation, with_channel
            # Simple test with pads.
            (self.test_context, [9, 9, 1],   [3, 3, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [9, 9, 1],   [3, 3, 1, 1], [1, 1, 1, 1], [2, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [9, 9, 1],   [3, 3, 1, 1], [1, 1, 1, 1], [0, 2, 0, 0], [1,1,1,1], False),
            (self.test_context, [9, 9, 1],   [3, 3, 1, 1], [1, 1, 1, 1], [0, 0, 2, 0], [1,1,1,1], False),
            (self.test_context, [9, 9, 1],   [3, 3, 1, 1], [1, 1, 1, 1], [0, 0, 0, 2], [1,1,1,1], False),
            (self.test_context, [9, 9, 1],   [3, 3, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [1,1,1,1], False),
            (self.test_context, [9, 9, 1],   [3, 3, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1,1,1,1], False),
            # Test with channels and padding.
            (self.test_context, [12, 12, 3], [6, 6, 3, 4], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [12, 12, 3], [6, 6, 3, 4], [1, 1, 1, 1], [5, 5, 5, 5], [1,1,1,1], False),
            # Test with stride, padding, and channels.
            (self.test_context, [13, 13, 3], [6, 6, 3, 1], [1, 2, 1, 1], [4, 4, 4, 4], [1,1,1,1], False),
            (self.test_context, [13, 13, 3], [2, 2, 3, 1], [1, 1, 2, 1], [1, 1, 1, 1], [1,1,1,1], False),
            #   The padded image is not evenly divisible by the stride.
            (self.test_context, [13, 13, 3], [5, 5, 3, 1], [1, 2, 1, 1], [3, 3, 3, 3], [1,1,1,1], False),
            (self.test_context, [13, 13, 3], [5, 5, 3, 1], [1, 1, 2, 1], [3, 3, 3, 3], [1,1,1,1], False),
            # Test with dialations.
            (self.test_context, [10, 10, 3], [4, 4, 3, 2], [1, 1, 1, 1], [1, 1, 1, 1], [1,2,2,1], False),
            (self.test_context, [13, 13, 3], [5, 5, 3, 2], [1, 2, 2, 1], [1, 1, 1, 1], [1,2,2,1], False),
            # Test with filter in_channels != image in_channels.
            (self.test_context, [13, 13, 3], [5, 5, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], True),
            (self.test_context, [13, 13, 3], [5, 5, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1,2,2,1], True),
            # fmt: on
        ]
        for c in test_configs:
            with self.subTest(f"{self._testMethodName} with config `{c}`."):
                self._test_conv2d_pt_ct(c[0], c[1], c[2], c[3], c[4], c[5], c[6], False)

    def test_conv2d_transpose_pt_ct(self):
        test_configs = [
            # fmt: off
            # Context,          im_shape,  filter_shape, stride,       padding,      dilation, with_channel
            # Simple test.
            (self.test_context, [2, 2, 1], [2, 2, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [2, 2, 1], [3, 3, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [3, 3, 1], [2, 2, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], False),
            # With pads.
            (self.test_context, [3, 3, 1], [3, 3, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [3, 3, 1], [3, 3, 1, 1], [1, 1, 1, 1], [0, 1, 0, 0], [1,1,1,1], False),
            (self.test_context, [3, 3, 1], [3, 3, 1, 1], [1, 1, 1, 1], [0, 0, 1, 0], [1,1,1,1], False),
            (self.test_context, [3, 3, 1], [3, 3, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1], [1,1,1,1], False),
            # With channels and pads.
            (self.test_context, [2, 2, 3], [2, 2, 1, 3], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [2, 2, 3], [2, 2, 4, 3], [1, 1, 1, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [2, 2, 3], [2, 2, 1, 3], [1, 1, 1, 1], [1, 1, 1, 1], [1,1,1,1], False),
            (self.test_context, [2, 2, 3], [2, 2, 4, 3], [1, 1, 1, 1], [1, 1, 1, 1], [1,1,1,1], False),
            # With stride, channels, and pads.
            (self.test_context, [2, 2, 1], [3, 3, 1, 1], [1, 2, 2, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [4, 4, 1], [4, 4, 1, 1], [1, 3, 3, 1], [0, 0, 0, 0], [1,1,1,1], False),
            (self.test_context, [4, 4, 3], [4, 4, 1, 3], [1, 3, 3, 1], [2, 2, 2, 2], [1,1,1,1], False),
            # fmt: on
        ]
        for c in test_configs:
            with self.subTest(f"{self._testMethodName} with config `{c}`."):
                self._test_conv2d_pt_ct(c[0], c[1], c[2], c[3], c[4], c[5], c[6], True)


if __name__ == "__main__":
    tf.test.main()
