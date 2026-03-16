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
from multiprocessing import Pool
from itertools import repeat


class TestShellTensorFastRotation(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        # TensorFlow only supports rotation for int32, int64, float32, and float64 types.
        # Create test contexts for all integer datatypes.
        cls.test_contexts = []

        for int_dtype in [tf.int32, tf.int64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 3],
                    plaintext_dtype=int_dtype,
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=1,
                )
            )

        # Create test contexts for floating point datatypes at a real-world
        # scaling factor.
        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033, 12289],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=8,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def _test_fast_reduce_sum_axis_0(self, test_context):
        # reduce_sum across axis 0 requires adding over all the slots.
        tftensor = test_utils.uniform_for_n_adds(
            test_context, num_adds=test_context.shell_context.num_slots / 2
        )

        s = hadal.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = hadal.to_encrypted(s, test_context.key)

        enc_reduce_sum = hadal.fast_reduce_sum(enc)

        tftensor_out = hadal.to_tensorflow(
            enc_reduce_sum, key=test_context.fast_rotation_key
        )
        self.assertAllClose(tftensor_out, hadal.reduce_sum(tftensor, axis=0), atol=1e-3)

    def test_fast_reduce_sum_axis_0(self):
        for test_context in self.test_contexts:
            with self.subTest(f"fast_reduce_sum_axis_0 with context {test_context}"):
                self._test_fast_reduce_sum_axis_0(test_context)

    def test_decrypt_with_wrong_key(self):
        test_context = self.test_contexts[0]
        tftensor = tf.ones(
            [test_context.shell_context.num_slots], dtype=test_context.plaintext_dtype
        )
        enc = hadal.to_encrypted(tftensor, test_context.key, test_context.shell_context)

        enc_reduce_sum = hadal.fast_reduce_sum(enc)
        wrong_key = test_context.key

        self.assertRaisesWithLiteralMatch(
            ValueError,
            "ShellFastRotationKey must be provided to decrypt a fast-rotated ShellTensor.",
            hadal.to_tensorflow,
            enc_reduce_sum,
            key=wrong_key,
        )

    def test_no_fast_reduce_sum_degree_two_ct(self):
        test_context = self.test_contexts[0]
        tftensor = tf.ones(
            [test_context.shell_context.num_slots, 1],
            dtype=test_context.plaintext_dtype,
        )

        enc = hadal.to_encrypted(tftensor, test_context.key, test_context.shell_context)
        enc_degree_two = enc * enc

        with self.assertRaisesOpError("Only Degree 1 ciphertexts supported."):
            hadal.fast_reduce_sum(enc_degree_two)

    def test_no_ctct_mul_after_fast_reduce_sum(self):
        test_context = self.test_contexts[0]
        tftensor = tf.ones(
            [test_context.shell_context.num_slots], dtype=test_context.plaintext_dtype
        )
        enc = hadal.to_encrypted(tftensor, test_context.key, test_context.shell_context)

        enc_reduce_sum = hadal.fast_reduce_sum(enc)

        self.assertRaisesWithLiteralMatch(
            ValueError,
            "A ShellTensor which has been fast-rotated or fast-reduced-summed cannot be multiplied with another ciphertext.",
            enc_reduce_sum.__mul__,
            enc,
        )

        self.assertRaisesWithLiteralMatch(
            ValueError,
            "A ShellTensor which has been fast-rotated or fast-reduced-summed cannot be multiplied with another ciphertext.",
            enc.__mul__,
            enc_reduce_sum,
        )


if __name__ == "__main__":
    tf.test.main()
