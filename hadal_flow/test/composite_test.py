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

        # Create test contexts for all integer datatypes. While this test
        # performs at most two multiplications, since the scaling factor is 1
        # there are no restrictions on the main moduli.
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

        # Create test contexts for floating point datatypes at various scaling
        # factors. Since these test perform at two multiplications, the trailing
        # two main moduli should be roughly equal to the scaling factor, for
        # modulus reduction during each multiplication.
        for float_dtype in [tf.float32, tf.float64]:
            cls.test_contexts.append(
                test_utils.TestContext(
                    outer_shape=[3, 2, 3],
                    plaintext_dtype=float_dtype,
                    # Num plaintext bits: 40, noise bits: 50
                    # Max plaintext value: 31, est error: 1.561%
                    log_n=11,
                    main_moduli=[288230376151748609, 2147565569, 147457, 114689],
                    aux_moduli=[],
                    plaintext_modulus=1099511795713,
                    scaling_factor=131073,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def _test_ct_ct_mulmul(self, test_context):
        try:
            # This test performs two multiplications.
            a = test_utils.uniform_for_n_muls(test_context, num_muls=2)
            b = test_utils.uniform_for_n_muls(test_context, num_muls=2)
        except Exception as e:
            print(
                f"Note: Skipping test ct_ct_mulmul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        ea = hadal.to_encrypted(a, test_context.key, test_context.shell_context)
        eb = hadal.to_encrypted(b, test_context.key, test_context.shell_context)

        ec = ea * eb
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key), atol=1e-3)

        # Here, ec has a mul depth of 1 while eb has a mul depth of 0. To
        # multiply them, hadal needs to account for the difference in scaling
        # factors. For ct_ct multiplication, the scaling factors must match, so
        # in this case eb will be scaled up with a ct_pt multiplication to match
        # ec. hadal will handle this automatically.
        ed = ec * eb
        self.assertEqual(ed._scaling_factor, (ea._scaling_factor**2) ** 2)
        self.assertAllClose(
            a * b * b, hadal.to_tensorflow(ed, test_context.key), atol=1e-3
        )

        # Make sure the original ciphertexts are not modified.
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key), atol=1e-3)
        self.assertAllClose(b, hadal.to_tensorflow(eb, test_context.key), atol=1e-3)

    def test_ct_ct_mulmul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_ct_mulmul with context {test_context}"):
                self._test_ct_ct_mulmul(test_context)

    def _test_ct_pt_mulmul(self, test_context):
        try:
            # This test performs two multiplications.
            a = test_utils.uniform_for_n_muls(test_context, num_muls=2)
            b = test_utils.uniform_for_n_muls(test_context, num_muls=2)
        except Exception as e:
            print(
                f"Note: Skipping test ct_pt_mulmul with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        ea = hadal.to_encrypted(a, test_context.key, test_context.shell_context)
        eb = hadal.to_encrypted(b, test_context.key, test_context.shell_context)

        ec = ea * eb
        self.assertAllClose(a * b, hadal.to_tensorflow(ec, test_context.key), atol=1e-3)

        # Here, ec has a mul depth of 1 while b is has a mul depth of 0. To
        # multiply them, hadal needs to account for the difference in
        # scaling factors. For ct_pt multiplication, the scaling factors do
        # not need to match, but their product must be remembered and divided
        # out when the result is decrypted. hadal will handle this
        # automatically.
        ed = ec * b
        self.assertEqual(ed._scaling_factor, ea._scaling_factor**3)
        self.assertAllClose(
            a * b * b, hadal.to_tensorflow(ed, test_context.key), atol=1e-3
        )

        # Make sure the original ciphertexts are not modified.
        self.assertAllClose(a, hadal.to_tensorflow(ea, test_context.key), atol=1e-3)
        self.assertAllClose(b, hadal.to_tensorflow(eb, test_context.key), atol=1e-3)

    def test_ct_pt_mulmul(self):
        for test_context in self.test_contexts:
            with self.subTest(f"ct_pt_mulmul with context {test_context}"):
                self._test_ct_pt_mulmul(test_context)


if __name__ == "__main__":
    tf.test.main()
