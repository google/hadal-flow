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
import tempfile


class TestShellContext(tf.test.TestCase):
    def test_key_save(self):
        # Num plaintext bits: 48, noise bits: 65
        # Max plaintext value: 127, est error: 3.840%
        context = hadal.create_context64(
            log_n=11,
            main_moduli=[288230376151748609, 18014398509506561],
            plaintext_modulus=281474976768001,
            scaling_factor=1052673,
        )
        key_path = tempfile.mkdtemp()  # Every trace gets a new key.
        key = hadal.create_key64(context, False, key_path)

        a = tf.ones([2**11, 2, 3], dtype=tf.float32) * 10
        ea = hadal.to_encrypted(a, key, context)

        # Try decrypting with the cached key
        cached_key = hadal.create_key64(context, True, key_path)
        self.assertAllClose(a, hadal.to_tensorflow(ea, cached_key))


if __name__ == "__main__":
    tf.test.main()
