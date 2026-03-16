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
import os
import time
import glob
import shutil

job_prefix = "tfshell_fs"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"
features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"
shared_dir = "/tmp/hadal_shared"
os.environ["HADAL_FILESYSTEM_PATH"] = shared_dir


class TestFilesystemDistributed(tf.test.TestCase):
    job_name = None

    def setUp(self):
        super().setUp()
        tf.compat.v1.disable_eager_execution()
        # Clean shared dir (only parent/features party should do this ideally, or check existence)
        if self.job_name == features_party_job:
            if os.path.exists(shared_dir):
                shutil.rmtree(shared_dir)
            os.makedirs(shared_dir, exist_ok=True)

    def test_distribution(self):
        config = tf.compat.v1.ConfigProto()
        config.graph_options.rewrite_options.custom_optimizers.add().name = (
            "FilesystemDistributionOptimizer"
        )

        cluster = tf.train.ClusterSpec(
            {
                f"{features_party_job}": ["localhost:3232"],
                f"{labels_party_job}": ["localhost:3233"],
            }
        )

        server = tf.distribute.Server(
            cluster, job_name=self.job_name, task_index=0, config=config
        )

        if self.job_name == labels_party_job:
            server.join()
            return

        # Features party
        time.sleep(2)

        with tf.compat.v1.Session(server.target, config=config) as sess:

            shell_context = hadal.create_context64(
                log_n=11,
                main_moduli=[8556589057, 8388812801],
                aux_moduli=[],
                plaintext_modulus=40961,
                scaling_factor=10,
            )

            # Compute decrypt(c = encrypt(a) + b)
            with tf.device(labels_party_dev):
                key = hadal.create_key64(shell_context)
                a = tf.random.uniform(
                    [shell_context.num_slots, 3], dtype=tf.float32, maxval=10
                )
                enc_a = hadal.to_encrypted(a, key, shell_context)

            with tf.device(features_party_dev):
                b = tf.random.uniform(
                    [shell_context.num_slots, 3], dtype=tf.float32, maxval=10
                )
                # enc_a is sent from labels -> features.
                enc_c = enc_a + b

            with tf.device(labels_party_dev):
                # enc_c is sent from features -> labels
                c = hadal.to_tensorflow(enc_c, key)

            # Run
            c_val, a_val, b_val = sess.run([c, a, b])

            self.assertAllClose(c_val, a_val + b_val, atol=1)

            # Verify files
            files = glob.glob(f"{shared_dir}/*.bin")
            print(f"Files created: {files}")
            self.assertNotEmpty(files)

            # Cleanup
            shutil.rmtree(shared_dir)
            os.makedirs(shared_dir, exist_ok=True)

            # Compute decode(z = encode(x) * y)
            with tf.device(labels_party_dev):
                x = tf.random.uniform(
                    [shell_context.num_slots, 3], dtype=tf.float32, maxval=10
                )
                pt_x = hadal.to_shell_plaintext(x, shell_context)

            with tf.device(features_party_dev):
                y = tf.random.uniform(
                    [shell_context.num_slots, 3], dtype=tf.float32, maxval=10
                )
                # pt_x is sent from labels -> features
                pt_z = pt_x * y

            with tf.device(labels_party_dev):
                # pt_z is sent from features -> labels
                z = hadal.to_tensorflow(pt_z)

            x_val, y_val, z_val = sess.run([x, y, z])
            self.assertAllClose(z_val, x_val * y_val, atol=1)

            files = glob.glob(f"{shared_dir}/*.bin")
            print(f"Files created test: {files}")
            self.assertNotEmpty(files)


if __name__ == "__main__":
    pid = os.fork()
    if pid == 0:  # child process
        TestFilesystemDistributed.job_name = labels_party_job
        tf.test.main()
        os._exit(0)
    else:  # parent process
        TestFilesystemDistributed.job_name = features_party_job
        tf.test.main()
        os.waitpid(pid, 0)
