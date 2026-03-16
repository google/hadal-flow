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
        # Create test contexts used for all tests.
        cls.test_contexts = []

        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[],
                    plaintext_dtype=float_dtype,
                    log_n=11,
                    main_moduli=[8556589057, 8388812801],
                    aux_moduli=[],
                    plaintext_modulus=40961,
                    scaling_factor=5,
                )
            )

    def checkScalingFactor(
        self, shell_tensor, expected_scaling_factor, expected_val, key=None
    ):
        # First check the scaling factor is as expected.
        self.assertEqual(shell_tensor.scaling_factor, expected_scaling_factor)

        # Then spoof the scaling factor to 1 and decode back to tf tensor.
        spoofed = hadal.ShellTensor64(
            _raw_tensor=shell_tensor._raw_tensor,
            _context=shell_tensor._context,
            _level=shell_tensor._level,
            _num_mod_reductions=shell_tensor._num_mod_reductions,
            _underlying_dtype=shell_tensor._underlying_dtype,
            _scaling_factor=1.0,
            _is_enc=shell_tensor._is_enc,
            _is_fast_rotated=shell_tensor._is_fast_rotated,
        )
        tf_spoofed_out = hadal.to_tensorflow(spoofed, key)
        self.assertAllClose(tf_spoofed_out[0], expected_val * expected_scaling_factor)

        # Last decode the original tensor and check it matches the input.
        tf_tensor_out = hadal.to_tensorflow(shell_tensor, key)
        self.assertAllClose(tf_tensor_out[0], expected_val)

    def _test_initial_scaling_factor(self, test_context):
        tf_tensor = [1.0]
        shell_tensor = hadal.to_shell_plaintext(tf_tensor, test_context.shell_context)
        self.checkScalingFactor(
            shell_tensor, test_context.shell_context.scaling_factor, 1.0
        )

        enc = hadal.to_encrypted(shell_tensor, test_context.key)
        self.checkScalingFactor(
            enc, test_context.shell_context.scaling_factor, 1.0, key=test_context.key
        )

    def test_initial_scaling_factor(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_initial_scaling_factor(test_context)

    def _test_add_scaling_factor(self, test_context):
        # The values below assume the scaling factor is 5.
        self.assertEqual(test_context.shell_context.scaling_factor, 5)
        a = [0.2]
        b = [0.8]

        pa = hadal.to_shell_plaintext(a, test_context.shell_context)
        pc = pa + b
        self.checkScalingFactor(pc, test_context.shell_context.scaling_factor, 1.0)

        ea = hadal.to_encrypted(pa, test_context.key)
        ec = ea + b
        self.checkScalingFactor(
            ec, test_context.shell_context.scaling_factor, 1.0, key=test_context.key
        )

    def test_add_scaling_factor(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_add_scaling_factor(test_context)

    def _test_mul_scaling_factor(self, test_context):
        # The values below assume the scaling factor is 5.
        self.assertEqual(test_context.shell_context.scaling_factor, 5)
        a = [0.2]
        b = [0.8]

        pa = hadal.to_shell_plaintext(a, test_context.shell_context)
        pc = pa * b
        self.checkScalingFactor(pc, test_context.shell_context.scaling_factor**2, 0.16)

        ea = hadal.to_encrypted(pa, test_context.key)
        ec = ea * b
        self.checkScalingFactor(
            ec, test_context.shell_context.scaling_factor**2, 0.16, key=test_context.key
        )

    def test_mul_scaling_factor(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_mul_scaling_factor(test_context)


if __name__ == "__main__":
    tf.test.main()
