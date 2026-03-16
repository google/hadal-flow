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
                    # Num plaintext bits: 22, noise bits: 36
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=1,
                )
            )

        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    # Num plaintext bits: 22, noise bits: 36
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=8,
                )
            )

        # Test with empty outer shape.
        cls.test_contexts.append(
            test_utils.TestContext(
                outer_shape=[],
                plaintext_dtype=tf.int32,
                # Num plaintext bits: 22, noise bits: 36
                # Max representable value: 65728
                log_n=11,
                main_moduli=[144115188076060673],
                aux_moduli=[],
                plaintext_modulus=4206593,
                scaling_factor=1,
            )
        )

        cls.test_contexts.append(
            test_utils.TestContext(
                outer_shape=[1],
                plaintext_dtype=tf.int32,
                # Num plaintext bits: 22, noise bits: 36
                # Max representable value: 65728
                log_n=11,
                main_moduli=[144115188076060673],
                aux_moduli=[],
                plaintext_modulus=4206593,
                scaling_factor=1,
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def _test_ct_ct_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        sb = hadal.to_shell_plaintext(b, test_context.shell_context)
        ea = hadal.to_encrypted(sa, test_context.key)
        eb = hadal.to_encrypted(sb, test_context.key)

        ec = ea * eb
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key))

        # Make sure the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa))
        self.assertAllClose(b, hadal.to_tensorflow(sb))
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))
        self.assertAllClose(b, hadal.to_tensorflow(eb, test_context.key))

    def test_ct_ct_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_ct_ct_mul(test_context)

    def _test_ct_ct_mul_with_broadcast(self, test_context):
        if len(test_context.outer_shape) < 2:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough dimensions to support this test."
            )
            return

        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        sb = hadal.to_shell_plaintext(b, test_context.shell_context)
        ea = hadal.to_encrypted(sa, test_context.key)
        eb = hadal.to_encrypted(sb, test_context.key)

        ec = ea * eb
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key))

        # Make sure the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa))
        self.assertAllClose(b, hadal.to_tensorflow(sb))
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))
        self.assertAllClose(b, hadal.to_tensorflow(eb, test_context.key))

    def test_ct_ct_mul_with_broadcast(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_ct_ct_mul_with_broadcast(test_context)

    def _test_ct_pt_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        sb = hadal.to_shell_plaintext(b, test_context.shell_context)
        ea = hadal.to_encrypted(sa, test_context.key)

        ec = ea * sb
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key))

        ed = sb * ea
        self.assertAllClose(a * b, hadal.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa))
        self.assertAllClose(b, hadal.to_tensorflow(sb))
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))

    def test_ct_pt_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_ct_pt_mul(test_context)

    def _test_ct_pt_mul_with_broadcast(self, test_context):
        if len(test_context.outer_shape) < 2:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough dimensions to support this test."
            )
            return

        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        sb = hadal.to_shell_plaintext(b, test_context.shell_context)
        ea = hadal.to_encrypted(sa, test_context.key)

        ec = ea * sb
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key))

        ed = sb * ea
        self.assertAllClose(a * b, hadal.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa))
        self.assertAllClose(b, hadal.to_tensorflow(sb))
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))

    def test_ct_pt_mul_with_broadcast(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_ct_pt_mul_with_broadcast(test_context)

    def _test_ct_tf_scalar_mul(self, test_context, scalar_shape):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)

            # Choose a random scalar to add to a.
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=scalar_shape)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = hadal.to_encrypted(a, test_context.key, test_context.shell_context)
        c = a * b
        ec = ea * b

        # The shell tensor should have the same shape as the TensorFlow Tensor.
        self.assertEqual(c.shape, ec.shape)
        # The values should also be the same.
        self.assertAllClose(c, hadal.to_tensorflow(ec, test_context.key))

        d = b * a
        ed = b * ea
        # The shell tensor should have the same shape as the TensorFlow Tensor.
        self.assertEqual(d.shape, ed.shape)
        # The values should also be the same.
        self.assertAllClose(d, hadal.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))

    def test_ct_tf_scalar_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                for scalar_shape in [[], [1], [1] + test_context.outer_shape]:
                    self._test_ct_tf_scalar_mul(test_context, scalar_shape)

    def _test_pt_tf_scalar_mul(self, test_context, scalar_shape):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)

            # Choose a random scalar to add to a.
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=scalar_shape)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        c = a * b
        sc = sa * b

        # The shell tensor should have the same shape as the TensorFlow Tensor.
        self.assertEqual(c.shape, sc.shape)
        # The values should also be the same.
        self.assertAllClose(c, hadal.to_tensorflow(sc))

        d = b * a
        sd = b * sa
        # The shell tensor should have the same shape as the TensorFlow Tensor.
        self.assertEqual(d.shape, sd.shape)
        # The values should also be the same.
        self.assertAllClose(d, hadal.to_tensorflow(sd))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa))

    def test_pt_tf_scalar_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                for scalar_shape in [[], [1], [1] + test_context.outer_shape]:
                    self._test_pt_tf_scalar_mul(test_context, scalar_shape)

    def _test_ct_tf_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        ea = hadal.to_encrypted(sa, test_context.key)

        ec = ea * b
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, hadal.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))

    def test_ct_tf_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_ct_tf_mul(test_context)

    def _test_ct_tf_mul_with_broadcast(self, test_context):
        if len(test_context.outer_shape) < 2:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough dimensions to support this test."
            )
            return

        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            # Set the last two dimensions to 1 to test broadcasting.
            b_shape = a.shape[:-2] + (1, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=b_shape)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        ea = hadal.to_encrypted(sa, test_context.key)

        ec = ea * b
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key))

        ed = b * ea
        self.assertAllClose(a * b, hadal.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))

    def test_ct_tf_mul_with_broadcast(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_ct_tf_mul_with_broadcast(test_context)

    def _test_ct_list_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        # Convert b to a python list where the first dimension is only 2.
        # When encoded, this will be padded to match the first dimension of a,
        # i.e. the number of slots in the ciphertext.
        b_list = b[:2, ...].numpy().tolist()

        # Set the unused elements of b to 0 for checking purposes.
        b = tf.concat([b[:2, ...], tf.zeros_like(b[2:, ...])], axis=0)

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        ea = hadal.to_encrypted(sa, test_context.key)

        ec = ea * b_list
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key))

        ed = b_list * ea
        self.assertAllClose(a * b, hadal.to_tensorflow(ed, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))

    def test_ct_list_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_ct_list_mul(test_context)

    def _test_pt_pt_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)
        sb = hadal.to_shell_plaintext(b, test_context.shell_context)

        sc = sa * sb
        self.assertAllClose(a * b, hadal.to_tensorflow(sc))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa))
        self.assertAllClose(b, hadal.to_tensorflow(sb))

    def test_pt_pt_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_pt_pt_mul(test_context)

    def _test_pt_tf_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)

        sc = sa * b
        self.assertAllClose(a * b, hadal.to_tensorflow(sc))

        sd = b * sa
        self.assertAllClose(a * b, hadal.to_tensorflow(sd))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa))

    def test_pt_tf_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_pt_tf_mul(test_context)

    def _test_pt_list_mul(self, test_context):
        try:
            # This test performs one multiplication.
            a = test_utils.uniform_for_n_muls(test_context, 1)
            b = test_utils.uniform_for_n_muls(test_context, 1)
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        # Convert b to a python list where the first dimension is only 2.
        # When encoded, this will be padded to match the first dimension of a,
        # i.e. the number of slots in the ciphertext.
        b_list = b[:2, ...].numpy().tolist()

        # Set the unused elements of b to 0 for checking purposes.
        b = tf.concat([b[:2, ...], tf.zeros_like(b[2:, ...])], axis=0)

        sa = hadal.to_shell_plaintext(a, test_context.shell_context)

        sc = sa * b_list
        self.assertAllClose(a * b, hadal.to_tensorflow(sc, test_context.key))

        sd = b_list * sa
        self.assertAllClose(a * b, hadal.to_tensorflow(sd, test_context.key))

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(sa, test_context.key))

    def test_pt_list_mul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                self._test_pt_list_mul(test_context)


if __name__ == "__main__":
    tf.test.main()
