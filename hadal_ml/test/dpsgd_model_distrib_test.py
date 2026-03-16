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
# import unittest
import os
import tensorflow as tf
import tempfile
import test_models

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"
features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"


class TestDistribModel(tf.test.TestCase):
    job_name = None

    def test_model(self):
        print(f"Job name: {self.job_name}")
        # flush stdout
        import sys

        sys.stdout.flush()

        cluster = tf.train.ClusterSpec(
            {
                f"{features_party_job}": ["localhost:2223"],
                f"{labels_party_job}": ["localhost:2224"],
            }
        )

        server = tf.distribute.Server(
            cluster,
            job_name=self.job_name,
            task_index=0,
        )

        tf.config.experimental_connect_to_cluster(cluster)

        # Register the hadal ops.
        import hadal
        import hadal_ml

        if self.job_name == labels_party_job:
            print(f"{self.job_name} server started.", flush=True)
            server.join()
            return

        crop_by = 12
        features_dataset, labels_dataset, val_dataset = test_models.MNIST_datasets(
            crop_by=crop_by,
            labels_party_dev=labels_party_dev,
            features_party_dev=features_party_dev,
        )

        inputs, outputs = test_models.MNIST_Shell_FF(
            10, inputs=(28 - crop_by, 28 - crop_by, 1)
        )

        with tf.device(features_party_dev):
            cache_dir = tempfile.TemporaryDirectory()
            cache = cache_dir.name

            m = hadal_ml.DpSgdModel(
                inputs=inputs,
                outputs=outputs,
                backprop_context_fn=lambda read_from_cache: hadal.create_autocontext64(
                    log2_cleartext_sz=33,
                    scaling_factor=16,
                    noise_offset_log2=50,
                    read_from_cache=read_from_cache,
                    cache_path=cache,
                ),
                noise_context_fn=lambda read_from_cache: hadal.create_autocontext64(
                    log2_cleartext_sz=33,
                    scaling_factor=1,
                    noise_offset_log2=48,
                    read_from_cache=read_from_cache,
                    cache_path=cache,
                ),
                labels_party_dev=labels_party_dev,
                features_party_dev=features_party_dev,
                cache_path=cache,
                check_overflow_INSECURE=True,
            )

            m.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(0.01, beta_1=0.8),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )

            m.summary()

        history = m.fit(
            features_dataset,
            labels_dataset,
            steps_per_epoch=4,
            epochs=1,
            verbose=2,
            validation_data=val_dataset,
        )

        cache_dir.cleanup()

        self.assertGreater(history.history["val_categorical_accuracy"][-1], 0.65)


if __name__ == "__main__":
    labels_pid = os.fork()
    if labels_pid == 0:  # child process
        TestDistribModel.job_name = labels_party_job
        tf.test.main()
        os._exit(0)

    TestDistribModel.job_name = features_party_job
    tf.test.main()

    os.waitpid(labels_pid, 0)
    print("Both parties finished.")
