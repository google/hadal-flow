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
        # Create test contexts used for all multiplication tests.
        int_dtypes = [
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.uint32,
            tf.int32,
            tf.uint64,
            tf.int64,
        ]
        cls.test_contexts = []

        for int_dtype in int_dtypes:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=int_dtype,
                    log_n=11,
                    main_moduli=[8556589057, 8388812801],
                    aux_moduli=[],
                    plaintext_modulus=40961,
                    scaling_factor=1,
                )
            )

        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    log_n=11,
                    main_moduli=[8556589057, 8388812801],
                    aux_moduli=[],
                    plaintext_modulus=40961,
                    scaling_factor=8,
                )
            )

        # Test with empty outer shape.
        cls.test_contexts.append(
            test_utils.TestContext(
                outer_shape=[],
                plaintext_dtype=tf.int32,
                log_n=11,
                main_moduli=[8556589057, 8388812801],
                aux_moduli=[],
                plaintext_modulus=40961,
                scaling_factor=1,
            )
        )

    def _test_create_shell_tensor(self, test_context):
        tf_tensor = test_utils.uniform_for_n_adds(test_context, 0)
        shell_tensor = hadal.to_shell_plaintext(tf_tensor, test_context.shell_context)
        tf_tensor_out = hadal.to_tensorflow(shell_tensor)
        self.assertAllClose(tf_tensor_out, tf_tensor)

    def test_create_shell_tensor(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_create_shell_tensor(test_context)

    def _test_encrypt_decrypt(self, test_context):
        tf_tensor = test_utils.uniform_for_n_adds(test_context, 0)

        # Encode the TensorFlow tensor to a shell plaintext.
        shell_tensor = hadal.to_shell_plaintext(tf_tensor, test_context.shell_context)
        tf_tensor_out = hadal.to_tensorflow(shell_tensor)
        self.assertAllClose(tf_tensor_out, tf_tensor)

        # Encrypt the shell plaintext.
        enc_shell_tensor = hadal.to_encrypted(shell_tensor, test_context.key)
        tf_tensor_dec_out = hadal.to_tensorflow(enc_shell_tensor, test_context.key)
        self.assertAllClose(tf_tensor_dec_out, tf_tensor)

        # Directly encrypt the tf tensor.
        enc_direct_shell_tensor = hadal.to_encrypted(
            tf_tensor, test_context.key, test_context.shell_context
        )
        tf_tensor_dec_direct_out = hadal.to_tensorflow(
            enc_direct_shell_tensor, test_context.key
        )
        self.assertAllClose(tf_tensor_dec_direct_out, tf_tensor)

    def test_encrypt_decrypt(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_encrypt_decrypt(test_context)

    def _test_shape(self, test_context):
        tf_tensor = tf.ones(
            [test_context.shell_context.num_slots] + test_context.outer_shape,
            dtype=tf.int32,
        )
        # Check the shape of a shell plaintext matches the cleartext.
        shell_tensor = hadal.to_shell_plaintext(tf_tensor, test_context.shell_context)
        self.assertAllClose(tf_tensor.shape, shell_tensor.shape)

        # Check the shape of an encryption matches the cleartext.
        enc_shell_tensor = hadal.to_encrypted(shell_tensor, test_context.key)
        self.assertAllClose(tf_tensor.shape, enc_shell_tensor.shape)

    def _test_shape(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_shape(test_context)

    def _test_shape_padding_python(self, test_context):
        context = test_context.shell_context
        key = test_context.key

        one_d = [1, 2]
        two_d = [[1, 2], [3, 4]]

        one_d = tf.cast(one_d, test_context.plaintext_dtype)
        two_d = tf.cast(two_d, test_context.plaintext_dtype)

        # Encode 1d tensor then check shape was padded and value is right.
        pt_1d = hadal.to_shell_plaintext(one_d, context)
        self.assertEqual(pt_1d.shape, context.num_slots)
        self.assertAllClose(one_d[:2], hadal.to_tensorflow(pt_1d)[:2])

        # Encode 2d tensor then check shape was padded and value is right.
        pt_2d = hadal.to_shell_plaintext(two_d, context)
        self.assertAllEqual(pt_2d.shape, tf.stack([context.num_slots, 2], axis=0))
        self.assertAllClose(two_d[:2][:2], hadal.to_tensorflow(pt_2d)[:2][:2])

        # Same tests as above but for encryption.
        ct_1d = hadal.to_encrypted(one_d, key, context)
        self.assertEqual(pt_1d.shape, context.num_slots)
        self.assertAllClose(one_d[:2], hadal.to_tensorflow(ct_1d, key)[:2])

        ct_2d = hadal.to_encrypted(two_d, key, context)
        self.assertAllEqual(pt_2d.shape, tf.stack([context.num_slots, 2], axis=0))
        self.assertAllClose(two_d[:2][:2], hadal.to_tensorflow(ct_2d, key)[:2][:2])

    def test_shape_padding_python(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_shape_padding_python(test_context)


if __name__ == "__main__":
    tf.test.main()
