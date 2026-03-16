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
import unittest
import tensorflow as tf
import keras
import numpy as np
import hadal_flow as hadal
import hadal_ml
import tempfile
import test_models


class TestModel(tf.test.TestCase):
    def _test_model(self, disable_encryption, disable_masking, disable_noise, cache):
        crop_by = 16
        features_dataset, labels_dataset, val_dataset = test_models.MNIST_datasets(
            crop_by=crop_by
        )

        inputs, outputs = test_models.MNIST_Simple_Shell_CNN(
            10, inputs=(28 - crop_by, 28 - crop_by, 1)
        )

        m = hadal_ml.DpSgdModel(
            inputs=inputs,
            outputs=outputs,
            backprop_context_fn=lambda read_from_cache: hadal.create_autocontext64(
                log2_cleartext_sz=29,
                scaling_factor=8,
                noise_offset_log2=0,
                read_from_cache=read_from_cache,
                cache_path=cache,
            ),
            noise_context_fn=lambda read_from_cache: hadal.create_autocontext64(
                log2_cleartext_sz=36,
                scaling_factor=1,
                noise_offset_log2=35,
                read_from_cache=read_from_cache,
                cache_path=cache,
            ),
            cache_path=cache,
            disable_he_backprop_INSECURE=disable_encryption,
            disable_masking_INSECURE=disable_masking,
            simple_noise_INSECURE=disable_noise,
            check_overflow_INSECURE=True,
            clip_threshold=10.0,
        )

        m.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.8),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        m.build(input_shape=(None, 28 - crop_by, 28 - crop_by, 1))
        m.summary()

        history = m.fit(
            features_dataset,
            labels_dataset,
            epochs=1,
            verbose=1,
            validation_data=val_dataset,
        )

        self.assertGreater(history.history["val_categorical_accuracy"][-1], 0.55)

    def test_model(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            self._test_model(False, False, False, cache_dir)


if __name__ == "__main__":
    unittest.main()
