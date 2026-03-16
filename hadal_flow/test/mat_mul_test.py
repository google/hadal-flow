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
        # Create test contexts used for matrix multiplication tests. TensorFlow
        # only supports matmul with int32, int64, float32, and float64.
        int_dtypes = [
            tf.int32,
            tf.int64,
        ]
        cls.test_contexts = []

        for int_dtype in int_dtypes:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=int_dtype,
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=1,
                    generate_rotation_keys=True,
                )
            )

        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    # Num plaintext bits: 22, noise bits: 64
                    # Max representable value: 65728
                    log_n=11,
                    main_moduli=[144115188076060673, 268460033],
                    aux_moduli=[],
                    plaintext_modulus=4206593,
                    scaling_factor=8,
                    generate_rotation_keys=True,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def _test_ct_tf_matmul(self, test_context):
        try:
            a = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[test_context.shell_context.num_slots, 5],
                subsequent_adds=5,  # For dim(1)
            )
            b = test_utils.uniform_for_n_muls(test_context, 1, shape=[5, 7])
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        ea = hadal.to_encrypted(a, test_context.key, test_context.shell_context)

        c = tf.matmul(a, b)

        @tf.function
        def test_functor():
            ec = hadal.matmul(ea, b)
            # Tests shape inference
            self.assertEqual(ec.shape.ndims, c.shape.ndims)
            for i in range(ec.shape.ndims):
                if ec.shape[i] is not None:
                    self.assertEqual(ec.shape[i], c.shape[i])
            return ec

        ec = test_functor()  # Run the core operation eagerly or lazily.

        self.assertAllClose(
            c,
            hadal.to_tensorflow(ec, test_context.key),
        )

        # Check the arguments were not modified.
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key))

    def test_ct_tf_matmul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"{self._testMethodName} with context `{test_context}`."):
                for eager in [False, True]:
                    with self.subTest(
                        f"{self._testMethodName} with context `{test_context}`, eager={eager}."
                    ):
                        tf.config.run_functions_eagerly(eager)
                        self._test_ct_tf_matmul(test_context)

    def _test_tf_ct_matmul(self, test_context, use_fast_rotation):
        # Generating the following tensors should always succeed since this test
        # uses it's own special context.
        try:
            a = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[3, 5, test_context.shell_context.num_slots],
                subsequent_adds=test_context.shell_context.num_slots / 2,
            )
            b = test_utils.uniform_for_n_muls(
                test_context,
                1,
                shape=[test_context.shell_context.num_slots, 2],
                subsequent_adds=test_context.shell_context.num_slots / 2,
            )
        except Exception as e:
            print(
                f"Note: Skipping test {self._testMethodName} with test context `{test_context}`. Not enough precision to support this test."
            )
            print(e)
            return

        eb = hadal.to_encrypted(b, test_context.key, test_context.shell_context)
        check_c = hadal.matmul(a, b, emulate_pt_ct=True)

        @tf.function
        def test_functor():
            if use_fast_rotation:
                ec = hadal.matmul(a, eb, pt_ct_reduction="fast")
            else:
                ec = hadal.matmul(a, eb, test_context.rotation_key)
            # Tests shape inference
            self.assertEqual(ec.shape.ndims, check_c.shape.ndims)
            for i in range(ec.shape.ndims):
                if ec.shape[i] is not None:
                    self.assertEqual(ec.shape[i], check_c.shape[i])
            return ec

        ec = test_functor()  # Run the core operation eagerly or lazily.

        if use_fast_rotation:
            dec_c = hadal.to_tensorflow(ec, test_context.fast_rotation_key)
        else:
            dec_c = hadal.to_tensorflow(ec, test_context.key)

        self.assertAllClose(check_c, dec_c)

        # Check the arguments were not modified.
        self.assertAllClose(b, hadal.to_tensorflow(eb, test_context.key))

    def test_tf_ct_matmul(self):
        for test_context in self.test_contexts:
            for use_fast_rotation in [False, True]:
                for eager in [False, True]:
                    with self.subTest(
                        f"{self._testMethodName} with context `{test_context}`, use_fast_rotation={use_fast_rotation}, eager={eager}."
                    ):
                        tf.config.run_functions_eagerly(eager)
                        self._test_tf_ct_matmul(test_context, use_fast_rotation)


if __name__ == "__main__":
    tf.test.main()
