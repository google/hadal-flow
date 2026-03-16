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
import shutil
import sys
import multiprocessing

# Define job names and devices
job_prefix = "tfshell_large"
features_party_job = f"{job_prefix}features"
labels_party_job = f"{job_prefix}labels"
features_party_dev = f"/job:{features_party_job}/replica:0/task:0/device:CPU:0"
labels_party_dev = f"/job:{labels_party_job}/replica:0/task:0/device:CPU:0"


def run_features_party(fs_path):
    """Runs the features party (server + client) in a separate process."""
    # Ensure environment is set for this process
    if fs_path:
        os.environ["HADAL_FILESYSTEM_PATH"] = fs_path
    elif "HADAL_FILESYSTEM_PATH" in os.environ:
        del os.environ["HADAL_FILESYSTEM_PATH"]

    tf.compat.v1.disable_eager_execution()
    hadal.enable_optimization()

    config = tf.compat.v1.ConfigProto()
    if fs_path:
        config.graph_options.rewrite_options.custom_optimizers.add().name = (
            "FilesystemDistributionOptimizer"
        )

    cluster = tf.train.ClusterSpec(
        {
            f"{features_party_job}": ["localhost:3234"],
            f"{labels_party_job}": ["localhost:3235"],
        }
    )

    # Start the server for the features party
    server = tf.distribute.Server(
        cluster,
        job_name=features_party_job,
        task_index=0,
        config=config,
    )

    # Re-define devices for this process context
    features_party_dev_local = features_party_dev
    labels_party_dev_local = labels_party_dev

    with tf.compat.v1.Session(server.target, config=config) as sess:
        num_ciphertexts = 100000

        shell_context = hadal.create_context64(
            log_n=11,
            main_moduli=[8556589057, 8388812801],
            aux_moduli=[],
            plaintext_modulus=40961,
            scaling_factor=10,
        )

        print(f"Generating large tensor with {num_ciphertexts} ciphertexts...")

        with tf.device(features_party_dev_local):
            key = hadal.create_key64(shell_context)
            # Hardcode slots to 2048 to avoid any Tensor shape issues
            # Shape must be [slots, batch] so that when slots are packed, we are left with [batch]
            pt_data = tf.random.uniform(
                [2048, num_ciphertexts], dtype=tf.float32, maxval=10
            )
            large_enc = hadal.to_encrypted(pt_data, key, shell_context)

        print("Graph construction: Defining transfer...")
        with tf.device(labels_party_dev_local):
            # Identity op on the other device forces transfer
            received_enc = tf.identity(large_enc)
            received_shape = hadal.shape(received_enc)

            # Use hadal.reduce_sum over axis 1 (the ciphertext batch dimension)
            reduced = hadal.reduce_sum(received_enc, axis=1)

        with tf.device(features_party_dev_local):
            final_result = hadal.to_tensorflow(reduced, key)

        print("Running session...")
        start_time = time.time()
        # Run shape too to verify
        res, r_shape = sess.run([final_result, received_shape])
        end_time = time.time()
        print(
            f"Transfer and computation successful! Time taken: {end_time - start_time:.2f}s"
        )
        print(f"Received shape: {r_shape}")

        # Verify shape
        if r_shape[1] != num_ciphertexts:
            print(
                f"Shape mismatch! Expected second dim to be {num_ciphertexts}, got {r_shape[1]}"
            )
            sys.exit(1)  # Exit with error on shape mismatch


class TestLargeTensor(tf.test.TestCase):
    job_name = None

    def setUp(self):
        super().setUp()
        tf.compat.v1.disable_eager_execution()

        # Only set up the shared directory if we are the features party (client/driver)
        # and filesystem distribution is enabled.
        self.fs_path = os.environ.get("HADAL_FILESYSTEM_PATH")
        if self.job_name == features_party_job and self.fs_path:
            if os.path.exists(self.fs_path):
                shutil.rmtree(self.fs_path)
            os.makedirs(self.fs_path, exist_ok=True)

    def test_large_transfer(self):
        if self.job_name == labels_party_job:
            # Should not happen as we fork below, but for safety
            return

        # Features party (Driver) - Test Runner Logic

        # Run the actual features party (server + client) in a separate process.
        # This isolates the test runner from crashes (e.g. segfaults in server/client).
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=run_features_party, args=(self.fs_path,))
        p.start()
        p.join()

        if self.fs_path:
            # Filesystem distribution enabled: Expect Success (0)
            if p.exitcode != 0:
                self.fail(f"Test failed with exit code {p.exitcode}. Expected success.")
        else:
            # Filesystem distribution disabled: Expect Failure/Crash (<0 or non-zero)
            # Typically returns -11 (SIGSEGV)
            if p.exitcode == 0:
                self.fail("Test succeeded via gRPC but was expected to crash/fail.")
            else:
                print(f"Process crashed/failed as expected with exit code {p.exitcode}")

        # Cleanup
        if self.fs_path and os.path.exists(self.fs_path):
            shutil.rmtree(self.fs_path)


if __name__ == "__main__":
    # Fork to create two processes (tasks)
    pid = os.fork()
    if pid == 0:  # child process
        TestLargeTensor.job_name = labels_party_job

        # Start the labels party server in this process
        tf.compat.v1.disable_eager_execution()
        hadal.enable_optimization()

        config = tf.compat.v1.ConfigProto()
        if os.environ.get("HADAL_FILESYSTEM_PATH"):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "FilesystemDistributionOptimizer"
            )

        cluster = tf.train.ClusterSpec(
            {
                f"{features_party_job}": ["localhost:3234"],
                f"{labels_party_job}": ["localhost:3235"],
            }
        )
        server = tf.distribute.Server(
            cluster,
            job_name=labels_party_job,
            task_index=0,
            config=config,
        )
        server.join()
        os._exit(0)
    else:  # parent process
        TestLargeTensor.job_name = features_party_job
        tf.test.main()
        os.waitpid(pid, 0)
