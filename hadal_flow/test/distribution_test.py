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
import test_utils

job_prefix = "tfshell"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"
features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"


class TestDistributed(tf.test.TestCase):
    job_name = None

    def test_distribution(self):
        cluster = tf.train.ClusterSpec(
            {
                f"{features_party_job}": ["localhost:3222"],
                f"{labels_party_job}": ["localhost:3223"],
            }
        )

        server = tf.distribute.Server(
            cluster,
            job_name=self.job_name,
            task_index=0,
        )

        cdf = tf.config.experimental.ClusterDeviceFilters()
        # The features party acts as the leader and schedules ops on the labels
        # party.
        cdf.set_device_filters(features_party_job, 0, [f"/job:{labels_party_job}"])
        # Prevent the labels party from scheduling ops on the features party.
        cdf.set_device_filters(labels_party_job, 0, [])

        tf.config.experimental_connect_to_cluster(cluster, cluster_device_filters=cdf)

        if self.job_name == labels_party_job:
            server.join()
            return

        shell_context = hadal.create_context64(
            log_n=11,
            main_moduli=[8556589057, 8388812801],
            aux_moduli=[],
            plaintext_modulus=40961,
            scaling_factor=1,
        )

        with tf.device(labels_party_dev):
            key = hadal.create_key64(shell_context)
            fast_rotation_key = hadal.create_fast_rotation_key64(shell_context, key)
            a = tf.random.uniform(
                [shell_context.num_slots, 3], dtype=tf.float32, maxval=10
            )
            enc_a = hadal.to_encrypted(a, key, shell_context)

            d = tf.random.uniform(
                [shell_context.num_slots, 3], dtype=tf.float32, maxval=10
            )
            pt_d = hadal.to_shell_plaintext(d, shell_context)

        # enc_a is sent between the parties.
        with tf.device(features_party_dev):
            b = tf.random.uniform(
                [shell_context.num_slots, 3], dtype=tf.float32, maxval=10
            )
            enc_c = enc_a + b

        # enc_a is cached on the features party.
        # pt_d is sent between the parties.
        with tf.device(features_party_dev):
            enc_e = enc_a + pt_d
            enc_f = hadal.fast_reduce_sum(enc_e)

        # enc_c is sent between the parties.
        with tf.device(labels_party_dev):
            c = hadal.to_tensorflow(enc_c, key)
            e = hadal.to_tensorflow(enc_e, key)
            f = hadal.to_tensorflow(enc_f, fast_rotation_key)

        self.assertAllClose(c, a + b, atol=1)
        self.assertAllClose(e, a + d, atol=1)
        self.assertAllClose(f, hadal.reduce_sum(a + d, axis=0), atol=1, rtol=1e-2)


if __name__ == "__main__":
    pid = os.fork()
    if pid == 0:  # child process
        TestDistributed.job_name = labels_party_job
        tf.test.main()
        os._exit(0)
    else:  # parent process
        TestDistributed.job_name = features_party_job
        tf.test.main()

        os.waitpid(pid, 0)
        print("Both parties finished.")
