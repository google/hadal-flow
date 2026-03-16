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


class TestShellTensorRotation(tf.test.TestCase):
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
                    generate_rotation_keys=True,
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
                    generate_rotation_keys=True,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def _test_roll(self, test_context, roll_num):
        # Create a tensor with the shape of slots x (outer_shape) where each
        # column of the first dimensions counts from 0 to slots-1. First check
        # if this tensor can actually be represented in the given context. A
        # large scaling factor would be what reduced the maximal representable
        # value.
        _, max_val = test_utils.get_bounds_for_n_adds(test_context, 0)
        if tf.cast(max_val, tf.int64) < test_context.shell_context.num_slots:
            print(
                f"Note: Skipping test roll with context {test_context}. Not enough precision to support this test. Context supports max val of {max_val} but need {test_context.shell_context.num_slots}."
            )
            return

        tftensor = tf.range(
            0,
            test_context.shell_context.num_slots,
            delta=1,
            dtype=test_context.plaintext_dtype,
        )
        # Expand dimensions to match the outer shape.
        for i in range(len(test_context.outer_shape)):
            tftensor = tf.expand_dims(tftensor, axis=-1)
            tftensor = tf.tile(
                tftensor, multiples=[1] * (i + 1) + [test_context.outer_shape[i]]
            )

        rolled_tftensor = hadal.roll(tftensor, roll_num)

        s = hadal.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = hadal.to_encrypted(s, test_context.key)

        rolled_enc = hadal.roll(enc, roll_num, test_context.rotation_key)
        rolled_result = hadal.to_tensorflow(rolled_enc, test_context.key)
        self.assertAllClose(rolled_tftensor, rolled_result, atol=1e-3)

    def test_roll(self):
        # Testing all contexts for all possible rotations is slow. Instead,
        # test a subset of rotations for each context, and one context tests
        # all rotations.
        for test_context in self.test_contexts:
            rotation_range = test_context.shell_context.num_slots // 2 - 1
            for roll_num in [-rotation_range, rotation_range, -1, 0, 1]:
                with self.subTest(
                    f"roll with context {test_context}, rotating by {roll_num}"
                ):
                    self._test_roll(test_context, roll_num)

        for test_context in [self.test_contexts[0]]:
            rotation_range = test_context.shell_context.num_slots // 2 - 1
            for roll_num in range(-rotation_range, rotation_range, 1):
                with self.subTest(
                    f"roll with context {test_context}, rotating by {roll_num}"
                ):
                    self._test_roll(test_context, roll_num)

    def _test_roll_mod_reduced(self, test_context, roll_num):
        # Create a tensor with the shape of slots x (outer_shape) where each
        # column of the first dimensions counts from 0 to slots-1. First check
        # if this tensor can actually be represented in the given context. A
        # large scaling factor would be what reduced the maximal representable
        # value.
        _, max_val = test_utils.get_bounds_for_n_adds(test_context, 0)
        if tf.cast(max_val, tf.int64) < test_context.shell_context.num_slots:
            print(
                f"Note: Skipping test roll with context {test_context}. Not enough precision to support this test. Context supports max val of {max_val} but need {test_context.shell_context.num_slots}."
            )
            return

        tftensor = tf.range(
            0,
            test_context.shell_context.num_slots,
            delta=1,
            dtype=test_context.plaintext_dtype,
        )
        # Expand dimensions to match the outer shape.
        for i in range(len(test_context.outer_shape)):
            tftensor = tf.expand_dims(tftensor, axis=-1)
            tftensor = tf.tile(
                tftensor, multiples=[1] * (i + 1) + [test_context.outer_shape[i]]
            )

        rolled_tftensor = hadal.roll(tftensor, roll_num)

        s = hadal.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = hadal.to_encrypted(s, test_context.key)

        # Test roll on a mod reduced ciphertext.
        enc_reduced = hadal.mod_reduce_tensor64(enc)
        rolled_enc_reduced = hadal.roll(
            enc_reduced, roll_num, test_context.rotation_key
        )
        rolled_result_reduced = hadal.to_tensorflow(
            rolled_enc_reduced, test_context.key
        )
        self.assertAllClose(rolled_tftensor, rolled_result_reduced, atol=1e-3)

    def test_roll_mod_reduced(self):
        # Testing all contexts for all possible rotations is slow. Instead,
        # test a subset of rotations for each context, and one context tests
        # all rotations.
        for test_context in self.test_contexts:
            rotation_range = test_context.shell_context.num_slots // 2 - 1
            for roll_num in [-rotation_range, rotation_range, -1, 0, 1]:
                with self.subTest(
                    f"roll_mod_reduced with context {test_context}, rotating by {roll_num}"
                ):
                    self._test_roll_mod_reduced(test_context, roll_num)

        for test_context in [self.test_contexts[0]]:
            rotation_range = test_context.shell_context.num_slots // 2 - 1
            for roll_num in range(-rotation_range, rotation_range, 1):
                with self.subTest(
                    f"roll_mod_reduced with context {test_context}, rotating by {roll_num}"
                ):
                    self._test_roll_mod_reduced(test_context, roll_num)

    def _test_reduce_sum_axis_0(self, test_context):
        # reduce_sum across axis 0 requires adding over all the slots.
        try:
            tftensor = test_utils.uniform_for_n_adds(
                test_context, num_adds=test_context.shell_context.num_slots / 2
            )
        except Exception as e:
            print(
                f"Note: Skipping test reduce_sum_axis_0 with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        s = hadal.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = hadal.to_encrypted(s, test_context.key)

        enc_reduce_sum = hadal.reduce_sum(
            enc, axis=0, rotation_key=test_context.rotation_key
        )

        tftensor_out = hadal.to_tensorflow(enc_reduce_sum, test_context.key)
        self.assertAllClose(tftensor_out, hadal.reduce_sum(tftensor, axis=0), atol=1e-3)

    def test_reduce_sum_axis_0(self):
        for test_context in self.test_contexts:
            with self.subTest(f"reduce_sum_axis_0 with context {test_context}"):
                self._test_reduce_sum_axis_0(test_context)

    def _test_reduce_sum_axis_n(self, test_context, outer_axis):
        # reduce_sum across `axis` requires adding over that dimension.
        try:
            num_adds = test_context.outer_shape[outer_axis]
            tftensor = test_utils.uniform_for_n_adds(test_context, num_adds)
        except Exception as e:
            print(
                f"Note: Skipping test reduce_sum_axis_n with context {test_context}. Not enough precision to support this test."
            )
            print(e)
            return

        s = hadal.to_shell_plaintext(tftensor, test_context.shell_context)
        enc = hadal.to_encrypted(s, test_context.key)

        enc_reduce_sum = hadal.reduce_sum(enc, axis=outer_axis + 1)

        tftensor_out = hadal.to_tensorflow(enc_reduce_sum, test_context.key)
        self.assertAllClose(
            tftensor_out, tf.reduce_sum(tftensor, axis=outer_axis + 1), atol=1e-3
        )

    def test_reduce_sum_axis_n(self):
        for test_context in self.test_contexts:
            for outer_axis in range(len(test_context.outer_shape)):
                with self.subTest(
                    f"reduce_sum_axis_n with context {test_context}, and axis {outer_axis+1}"
                ):
                    self._test_reduce_sum_axis_n(test_context, outer_axis)

    def _test_reduce_sum_with_modulus_pt(self, test_context, axis):
        t = tf.cast(
            test_context.shell_context.plaintext_modulus, test_context.plaintext_dtype
        )

        shape = [test_context.shell_context.num_slots] + test_context.outer_shape

        tftensor = tf.random.uniform(
            shape, minval=-t // 2, maxval=t // 2, dtype=test_context.plaintext_dtype
        )

        reduced = hadal.reduce_sum_with_mod(tftensor, axis, test_context.shell_context)

        # Manually compute the reduced sum with modulo after every addition to
        # avoid overflow. This includes introducing a scaling factor manually.
        unsigned = tf.where(tftensor < 0, t + tftensor, tftensor)
        unsigned = tf.cast(unsigned, tf.uint64)
        start_slice = [0] * len(unsigned.shape)
        size_slice = [-1] * len(unsigned.shape)
        size_slice[axis] = 1
        accum = tf.slice(unsigned, start_slice, size_slice)
        for i in range(1, shape[axis]):
            start_slice[axis] = i
            accum += tf.slice(unsigned, start_slice, size_slice)
            accum = tf.math.mod(accum, test_context.shell_context.plaintext_modulus)
        accum = tf.squeeze(accum)
        accum = tf.cast(accum, test_context.plaintext_dtype)
        accum = tf.where(accum > t // 2, accum - t, accum)

        self.assertAllClose(reduced, accum, atol=1e-3)

    def test_reduce_sum_with_modulus_pt(self):
        for test_context in self.test_contexts:
            for axis in range(len(test_context.outer_shape) + 1):
                with self.subTest(
                    f"{self._testMethodName} with context {test_context}, and axis {axis}"
                ):
                    self._test_reduce_sum_with_modulus_pt(test_context, axis)
                break
            break

    def test_mask_with_reduce_sum_modulus(self):
        context = hadal.create_context64(
            log_n=10,
            main_moduli=[8556589057, 8388812801],
            plaintext_modulus=40961,
            scaling_factor=3,
            seed="test_seed",
        )
        secret_key = hadal.create_key64(context)

        t = tf.cast(context.plaintext_modulus, tf.float32)
        t_half = t // 2
        s = tf.cast(context.scaling_factor, tf.float32)

        # The plaintexts must be be small enough to avoid overflow during reduce_sum on
        # axis 0, hence divide by the number of slots and scaling factor.
        max_pt = t / s / tf.cast(context.num_slots, tf.float32)
        tfdata = tf.random.uniform(
            [context.num_slots, 2], dtype=tf.float32, minval=-max_pt, maxval=max_pt
        )

        # The masks may overflow during the reduce_sum. When the masks are
        # operated on, they are multiplied by the scaling factor, so it is not
        # necessary to mask the full range -t/2 to t/2. (Though it is possible,
        # it unnecessarily introduces noise into the ciphertext.)
        mask = tf.random.uniform(
            hadal.shape(tfdata),
            dtype=tf.float32,
            minval=-t_half / s,
            maxval=t_half / s,
        )

        # Encrypt the data and mask.
        encdata = hadal.to_encrypted(tfdata, secret_key, context)
        masked_encdata = encdata + mask
        # Decrypt and reduce_sum the result.
        masked_dec = hadal.to_tensorflow(masked_encdata, secret_key)
        sum_masked_dec = hadal.reduce_sum_with_mod(masked_dec, 0, context)
        sum_mask = hadal.reduce_sum_with_mod(mask, 0, context)
        # Unmask and rebalance the result back into the range -t/2 to t/2
        # in the floating point domain.
        sum_dec = sum_masked_dec - sum_mask

        def rebalance(x, t_half):
            epsilon = tf.constant(1e-6, dtype=float)
            t_half_over_s = tf.cast(t_half / s, dtype=float)
            r_bound = t_half_over_s + epsilon
            l_bound = -t_half_over_s - epsilon
            t_over_s = tf.cast(t / s, dtype=float)
            x = tf.where(x > r_bound, x - t_over_s, x)
            x = tf.where(x < l_bound, x + t_over_s, x)
            return x

        sum_dec = rebalance(sum_dec, t_half)

        # The correct result after unmasking is simply the sum of the plaintexts
        # if the modulo operations were performed correctly.
        check = tf.reduce_sum(tfdata, axis=0)

        # The maximum error of this operation is the error of each individual
        # slot encoded (1/s/2) times the number of slots due to the reduce sum.
        atol = tf.cast(context.num_slots, dtype=float) * (1 / s / 2)

        self.assertAllClose(sum_dec, check, atol=atol)


if __name__ == "__main__":
    tf.test.main()
