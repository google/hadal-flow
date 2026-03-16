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

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


class TestShellTensor(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        cls.test_contexts = []
        cls.test_contexts.append(
            test_utils.TestContext(
                outer_shape=[3, 2, 3],
                plaintext_dtype=tf.float32,
                log_n=11,
                main_moduli=[8556589057, 8388812801],
                aux_moduli=[],
                plaintext_modulus=40961,
                scaling_factor=2,
            )
        )

    def _test_rand_round(self, test_context, value):
        tf_tensor = test_utils.uniform_for_n_adds(test_context, 0)
        tf_tensor = tf.ones_like(tf_tensor) * value

        # Encrypt the tf tensor.
        a = hadal.to_encrypted(tf_tensor, test_context.key, test_context.shell_context)
        b = hadal.to_tensorflow(a, test_context.key)

        # Check that the average value of b is around `value`.
        b_avg = tf.reduce_mean(b)
        self.assertAllClose(b_avg, value, atol=0.01)

        # Check that the values of b are not all the same.
        b_std = tf.math.reduce_std(b)
        self.assertGreater(b_std, 0.01)

    def test_rand_round(self):
        hadal.enable_randomized_rounding()

        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_rand_round(test_context, 0.1)
                self._test_rand_round(test_context, -0.1)
                self._test_rand_round(test_context, 2.1)
                self._test_rand_round(test_context, -2.1)


if __name__ == "__main__":
    tf.test.main()
