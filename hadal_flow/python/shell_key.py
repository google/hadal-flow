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
import hadal_flow.python.shell_ops as shell_ops
from hadal_flow.python.shell_context import ShellContext64
import tensorflow as tf
import typing


class ShellKey64(tf.experimental.ExtensionType):
    _raw_keys_at_level: tf.Tensor

    def _get_key_at_level(self, level):
        return self._raw_keys_at_level[level - 1]  # 0th level does not exist.


def create_key64(context, read_from_cache=False, cache_path=None):
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64")

    if read_from_cache and cache_path == None:
        raise ValueError(
            "A `cache_path` must be provided when `read_from_cache` is True."
        )

    with tf.name_scope("create_key64"):
        if cache_path is not None:
            key_path = cache_path + "/" + context.id_str + "_key"

        if read_from_cache:
            cached_keys = tf.io.read_file(key_path)
            cached_keys = tf.io.parse_tensor(cached_keys, out_type=tf.variant)
            return ShellKey64(_raw_keys_at_level=cached_keys)

        # Generate the keys.
        num_keys = context.level
        raw_keys = tf.TensorArray(
            tf.variant, size=context.level, clear_after_read=False
        )

        # Generate and store the first key in the last index.
        raw_keys = raw_keys.write(
            num_keys - 1, shell_ops.key_gen64(context._get_context_at_level(num_keys))
        )

        # Mod reduce to compute the remaining keys.
        raw_keys, _ = tf.while_loop(
            lambda ks, l: l > 1,
            lambda ks, l: (
                ks.write(
                    l - 2,
                    shell_ops.modulus_reduce_key64(
                        context._get_context_at_level(l), ks.read(l - 1)
                    ),
                ),
                l - 1,
            ),
            loop_vars=[raw_keys, num_keys],
            shape_invariants=[
                tf.TensorSpec(None, dtype=tf.variant),
                tf.TensorSpec([], dtype=tf.int32),
            ],
            parallel_iterations=1,
        )
        raw_keys = raw_keys.gather(tf.range(0, num_keys))

        if cache_path != None:
            tf.io.write_file(key_path, tf.io.serialize_tensor(raw_keys))

        return ShellKey64(_raw_keys_at_level=raw_keys)


class ShellRotationKey64(tf.experimental.ExtensionType):
    _raw_keys_at_level: tf.Tensor

    def _get_key_at_level(self, level):
        return self._raw_keys_at_level[level - 1]  # 0th level does not exist.


def create_rotation_key64(context, key, read_from_cache=False, cache_path=None):
    """Create rotation keys for any multiplicative depth of the given context.
    Rotation key contains keys to perform an arbitrary number of slot rotations.
    Since rotation key generation is expensive, the caller can choose to skip
    generating keys at levels (particular number of moduli) at which no
    rotations are required."""
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    if not isinstance(key, ShellKey64):
        raise ValueError("key must be a ShellKey64.")

    if read_from_cache and cache_path == None:
        raise ValueError(
            "A `cache_path` must be provided when `read_from_cache` is True."
        )

    with tf.name_scope("create_rotation_key64"):
        if cache_path is not None:
            key_path = cache_path + "/" + context.id_str + "_rotkey"

        if read_from_cache:
            cached_keys = tf.io.read_file(key_path)
            cached_keys = tf.io.parse_tensor(cached_keys, out_type=tf.variant)
            return ShellRotationKey64(_raw_keys_at_level=cached_keys)

        # Generate the keys.
        num_keys = context.level
        raw_keys = tf.TensorArray(
            tf.variant,
            size=num_keys,
            clear_after_read=False,
            infer_shape=False,
            element_shape=(),
        )

        # Start from the highest level.
        raw_keys, _ = tf.while_loop(
            lambda ks, l: l > 0,
            lambda ks, l: (
                ks.write(
                    l - 1,
                    shell_ops.rotation_key_gen64(
                        context._get_context_at_level(l), key._get_key_at_level(l)
                    ),
                ),
                l - 1,
            ),
            loop_vars=[raw_keys, num_keys],
            shape_invariants=[
                tf.TensorSpec(None, dtype=tf.variant),
                tf.TensorSpec([], dtype=tf.int32),
            ],
            parallel_iterations=1,
        )
        raw_keys = raw_keys.gather(tf.range(0, num_keys))

        if cache_path != None:
            tf.io.write_file(key_path, tf.io.serialize_tensor(raw_keys))

        return ShellRotationKey64(_raw_keys_at_level=raw_keys)


class ShellFastRotationKey64(tf.experimental.ExtensionType):
    _raw_keys_at_level: tf.Tensor

    def _get_key_at_level(self, level):
        return self._raw_keys_at_level[level - 1]  # 0th level does not exist.


def create_fast_rotation_key64(context, key, read_from_cache=False, cache_path=None):
    """Create fast rotation keys for any multiplicative depth of the given context.
    Rotation key contains keys *decrypt* a previously "fast" rotated ciphertext.
    These keys are much faster to generated than regular rotation keys, and
    they are not needed for the rotation operation itself, only for decryption.
    Fast rotation is only supported for degree 1 ciphertexts.
    """
    if not isinstance(context, ShellContext64):
        raise ValueError("context must be a ShellContext64.")

    if not isinstance(key, ShellKey64):
        raise ValueError("key must be a ShellKey64.")

    if read_from_cache and cache_path == None:
        raise ValueError(
            "A `cache_path` must be provided when `read_from_cache` is True."
        )

    with tf.name_scope("create_fast_rotation_key64"):
        if cache_path is not None:
            key_path = cache_path + "/" + context.id_str + "_fastrotkey"

        if read_from_cache:
            cached_keys = tf.io.read_file(key_path)
            cached_keys = tf.io.parse_tensor(cached_keys, out_type=tf.variant)
            return ShellFastRotationKey64(_raw_keys_at_level=cached_keys)

        # Generate the keys.
        num_keys = context.level
        raw_keys = tf.TensorArray(
            tf.variant,
            size=num_keys,
            clear_after_read=False,
            infer_shape=False,
            element_shape=(),
        )

        # Start from the highest level.
        raw_keys, _ = tf.while_loop(
            lambda ks, l: l > 0,
            lambda ks, l: (
                ks.write(
                    l - 1,
                    shell_ops.fast_rotation_key_gen64(
                        context._get_context_at_level(l), key._get_key_at_level(l)
                    ),
                ),
                l - 1,
            ),
            loop_vars=[raw_keys, num_keys],
            shape_invariants=[
                tf.TensorSpec(None, dtype=tf.variant),
                tf.TensorSpec([], dtype=tf.int32),
            ],
            parallel_iterations=1,
        )

        raw_keys = raw_keys.gather(tf.range(0, num_keys))

        if cache_path != None:
            tf.io.write_file(key_path, tf.io.serialize_tensor(raw_keys))

        return ShellFastRotationKey64(_raw_keys_at_level=raw_keys)
