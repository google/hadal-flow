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


class TestDG(tf.test.TestCase):
    context = None

    @classmethod
    def setUpClass(cls):
        cls.context = hadal.create_context64(
            log_n=11,
            main_moduli=[8556589057, 8388812801],
            aux_moduli=[],
            plaintext_modulus=40961,
            scaling_factor=1,
        )

    @classmethod
    def tearDownClass(cls):
        cls.test_contexts = None

    def test_shape_inf(self):
        p = hadal.DiscreteGaussianParams(max_scale=2000.0, base_scale=8.0)
        scale = 25.37
        num_samples = 32

        def inf_shape():
            a, b = hadal.sample_centered_gaussian_f(scale, p)
            samples = hadal.sample_centered_gaussian_l(self.context, num_samples, p)
            return a.shape, b.shape, samples.shape

        a_sh, b_sh, samp_sh = inf_shape()
        self.assertEqual(a_sh, [5])
        self.assertEqual(b_sh, [5])
        self.assertEqual(samp_sh, [num_samples, 5])

    def test_sample(self):
        p = hadal.DiscreteGaussianParams(max_scale=2000.0, base_scale=7.5)
        scale = 680

        self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            ".*`s_base` must be at least sqrt\(2\) times the smoothing parameter.*",
            lambda: hadal.sample_centered_gaussian_l(self.context, 1, p),
        )

    def test_sample(self):
        p = hadal.DiscreteGaussianParams(max_scale=2000.0, base_scale=7.6)
        scale = 680
        num_samples = 1000000

        a, b = hadal.sample_centered_gaussian_f(scale, p)

        samples_a = hadal.sample_centered_gaussian_l(self.context, num_samples, p)
        samples_b = hadal.sample_centered_gaussian_l(self.context, num_samples, p)

        a = tf.expand_dims(a, -1)
        b = tf.expand_dims(b, -1)

        dg_samps = tf.matmul(samples_a, a) + tf.matmul(samples_b, b)

        stddev = tf.math.reduce_std(tf.cast(dg_samps, dtype=tf.float32))
        self.assertGreater(stddev, scale)

        avg = tf.reduce_mean(dg_samps)
        self.assertLessEqual(avg, 1)


if __name__ == "__main__":
    tf.test.main()
