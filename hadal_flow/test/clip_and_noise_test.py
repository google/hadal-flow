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
from threading import Thread
import os
from dataclasses import dataclass
import math


@dataclass
class ClipAndNoiseTestContext:
    shape: list[int]
    dtype: tf.DType
    num_bits: int


class TestClipAndNoise(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        cls.test_contexts = []

        test_shapes = [[], [1], [2, 3], [4, 5, 6]]

        for shape in test_shapes:
            cls.test_contexts.append(
                ClipAndNoiseTestContext(
                    shape=shape,
                    dtype=tf.int32,
                    num_bits=13,
                )
            )

        for shape in test_shapes:
            cls.test_contexts.append(
                ClipAndNoiseTestContext(
                    shape=shape,
                    dtype=tf.int64,
                    num_bits=37,
                )
            )

        cls.test_contexts.append(
            ClipAndNoiseTestContext(
                shape=shape,
                dtype=tf.int32,
                num_bits=32,
            )
        )

        cls.test_contexts.append(
            ClipAndNoiseTestContext(
                shape=shape,
                dtype=tf.int64,
                num_bits=64,
            )
        )

    def _test_clip_and_noise(self, test_context):
        min_val = -(2 ** (test_context.num_bits - 1))
        max_val = 2 ** (test_context.num_bits - 1) - 1

        # The maximum value of all elements which will not overflow when
        # computing the L2 norm squared of the flattened gradient.
        max_grad = math.floor(math.sqrt(max_val) / test_context.num_bits)
        min_grad = -max_grad
        self.assertGreater(max_grad, 0)

        g = tf.cast(
            tf.random.uniform(
                test_context.shape,
                dtype=tf.int64,
                minval=min_grad,
                maxval=max_grad,
            ),
            dtype=test_context.dtype,
        )
        r = tf.cast(
            tf.random.uniform(
                test_context.shape,
                dtype=tf.int64,
                minval=min_val,
                maxval=max_val,
            ),
            dtype=test_context.dtype,
        )
        n = tf.cast(
            tf.random.uniform(
                test_context.shape,
                dtype=tf.int64,
                minval=0,
                maxval=8,
            ),
            dtype=test_context.dtype,
        )
        c = tf.constant(512, dtype=test_context.dtype)

        # Run the labels party in a separate process. Note EMP requires each
        # party to be run in a separate process.
        pid = os.fork()
        if pid == 0:  # child process
            hadal.clip_and_noise_labels_party(
                g + r,
                c,
                n,
                Bitwidth=test_context.num_bits,
                StartPort=5555,
                FeaturePartyHost="127.0.0.1",
            )
            os._exit(0)
        else:  # parent process
            clipped_noised_grad = hadal.clip_and_noise_features_party(
                r,
                Bitwidth=test_context.num_bits,
                StartPort=5555,
                LabelPartyHost="127.0.0.1",
            )
            os.waitpid(pid, 0)  # Wait for child process to finish.

        if tf.reduce_sum(g * g) > c:
            correct = c + n
        else:
            correct = g + n

        # Emulate overflow of 2's complement addition between `Bitwidth`
        # integers from when g + n is computed. Any overflow in the masking /
        # unmasking from g + r - r will cancel out.
        correct = tf.where(correct > max_val, min_val + (correct - max_val), correct)

        self.assertAllEqual(clipped_noised_grad, correct)

    def test_clip_and_noise(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_clip_and_noise(test_context)


if __name__ == "__main__":
    tf.test.main()
