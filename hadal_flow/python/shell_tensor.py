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
import math
import tensorflow as tf
import hadal_flow.python.shell_ops as shell_ops
from hadal_flow.python.shell_context import ShellContext64
from hadal_flow.python.shell_key import ShellKey64
from hadal_flow.python.shell_key import ShellRotationKey64
from hadal_flow.python.shell_key import ShellFastRotationKey64


class ShellTensor64(tf.experimental.ExtensionType):
    _raw_tensor: tf.Tensor
    _context: ShellContext64
    _level: tf.Tensor
    _num_mod_reductions: int
    _underlying_dtype: tf.DType
    _scaling_factor: float
    _is_enc: bool
    _is_fast_rotated: bool = False

    @property
    def shape(self):
        try:
            return tf.TensorShape([self._context.num_slots.numpy()]).concatenate(
                self._raw_tensor.get_shape()
            )
        except AttributeError:
            return tf.TensorShape([None]).concatenate(self._raw_tensor.get_shape())

    @property
    def ndim(self):
        return self._raw_tensor.ndim + 1

    @property
    def dtype(self):
        return tf.variant

    @property
    def name(self):
        return self._raw_tensor.name

    @property
    def plaintext_dtype(self):
        return self._underlying_dtype

    @property
    def is_encrypted(self):
        return self._is_enc

    @property
    def level(self):
        return self._level

    @property
    def scaling_factor(self):
        return self._scaling_factor

    def __getitem__(self, slice):
        slots = slice[0]
        if slots.start != None or slots.stop != None or slots.step != None:
            raise ValueError(
                f"ShellTensor does not support intra-slot slicing. Use `:` on the first dimension. Got {slice}"
            )
        return ShellTensor64(
            _raw_tensor=self._raw_tensor[slice[1:]],
            _context=self._context,
            _level=self._level,
            _num_mod_reductions=self._num_mod_reductions,
            _underlying_dtype=self._underlying_dtype,
            _scaling_factor=self._scaling_factor,
            _is_enc=self.is_encrypted,
            _is_fast_rotated=self._is_fast_rotated,
        )

    def __add__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli_and_scaling(self, other)

            if self.is_encrypted and other.is_encrypted:
                result_raw_tensor = shell_ops.add_ct_ct64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.add_ct_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif not self.is_encrypted and other.is_encrypted:
                result_raw_tensor = shell_ops.add_ct_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_other._raw_tensor,
                    matched_self._raw_tensor,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.add_pt_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                _raw_tensor=result_raw_tensor,
                _context=matched_self._context,
                _level=matched_self._level,
                _num_mod_reductions=matched_self._num_mod_reductions,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=matched_self._scaling_factor,
                _is_enc=self._is_enc or other._is_enc,
                _is_fast_rotated=self._is_fast_rotated or other._is_fast_rotated,
            )

        elif isinstance(other, tf.Tensor):
            if other.shape == () or other.shape == (1,):
                # In the special case of scalar addition, instead of padding
                # with zeros replicate the scalar across all slots and broadcast
                # to the correct shape.
                other = tf.broadcast_to(
                    other, tf.expand_dims(self._context.num_slots, 0)
                )

            elif other.shape[0] == 1 and len(other.shape) == len(self.shape):
                # In the special case of broadcasting over the packing'
                # dimension, replicate the scalar across all slots.
                other = tf.broadcast_to(
                    other,
                    tf.concat(
                        [tf.expand_dims(self._context.num_slots, 0), other.shape[1:]],
                        axis=0,
                    ),
                )

            # Lift tensorflow tensor to shell tensor with the same scaling
            # factor as self and attempt the addition again.
            so = to_shell_plaintext(other, self._context)
            return self + so

        else:
            # Try to import the unknown operand to a TensorFlow tensor and
            # attempt the subtraction again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(f"Unsupported type for addition. Got {type(other)}.")

            return self + tf_other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli_and_scaling(self, other)

            if self.is_encrypted and other.is_encrypted:
                result_raw_tensor = shell_ops.sub_ct_ct64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.sub_ct_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif not self.is_encrypted and other.is_encrypted:
                negative_other = -matched_other
                result_raw_tensor = shell_ops.add_ct_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    negative_other._raw_tensor,
                    matched_self._raw_tensor,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                result_raw_tensor = shell_ops.sub_pt_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                _raw_tensor=result_raw_tensor,
                _context=matched_self._context,
                _level=matched_self._level,
                _num_mod_reductions=matched_self._num_mod_reductions,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=matched_self._scaling_factor,
                _is_enc=self._is_enc or other._is_enc,
                _is_fast_rotated=self._is_fast_rotated or other._is_fast_rotated,
            )
        elif isinstance(other, tf.Tensor):
            if other.shape == () or other.shape == (1,):
                # In the special case of scalar subtraction, instead of padding
                # with zeros replicate the scalar across all slots and broadcast
                # to the correct shape.
                other = tf.broadcast_to(
                    other, tf.expand_dims(self._context.num_slots, 0)
                )

            elif other.shape[0] == 1 and len(other.shape) == len(self.shape):
                other = tf.broadcast_to(
                    other,
                    tf.concat(
                        [tf.expand_dims(self._context.num_slots, 0), other.shape[1:]],
                        axis=0,
                    ),
                )

            # Lift tensorflow tensor to shell tensor with the same scaling
            # factor as self and attempt the subtraction again.
            shell_other = to_shell_plaintext(other, self._context)
            return self - shell_other
        else:
            # Try to import the unknown operand to a TensorFlow tensor and
            # attempt the subtraction again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(
                    f"Unsupported type for subtraction. Got {type(other)}."
                )
            return self - tf_other

    def __rsub__(self, other):
        if isinstance(other, tf.Tensor):
            if other.shape == () or other.shape == (1,):
                # In the special case of scalar subtraction, instead of padding
                # with zeros replicate the scalar across all slots and broadcast
                # to the correct shape.
                other = tf.broadcast_to(
                    other, tf.expand_dims(self._context.num_slots, 0)
                )

            elif other.shape[0] == 1 and len(other.shape) == len(self.shape):
                other = tf.broadcast_to(
                    other,
                    tf.concat(
                        [tf.expand_dims(self._context.num_slots, 0), other.shape[1:]],
                        axis=0,
                    ),
                )

            # Import to a shell plaintext, which pads the first dimension with
            # zeros out to the number of slots.
            shell_other = to_shell_plaintext(other, self._context)

            if self.is_encrypted:
                negative_self = -self
                raw_result = shell_ops.add_ct_pt64(
                    self._context._get_context_at_level(self._level),
                    negative_self._raw_tensor,
                    shell_other._raw_tensor,
                )
            else:
                raw_result = shell_ops.sub_pt_pt64(
                    self._context._get_context_at_level(self._level),
                    shell_other._raw_tensor,
                    self._raw_tensor,
                )

            return ShellTensor64(
                _raw_tensor=raw_result,
                _context=self._context,
                _level=self._level,
                _num_mod_reductions=self._num_mod_reductions,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=self._scaling_factor,
                _is_enc=self._is_enc,
                _is_fast_rotated=self._is_fast_rotated,
            )
        else:
            # Try to import the unknown operand to a TensorFlow tensor and
            # attempt the the rsub again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(
                    f"Unsupported type for subtraction. Got {type(other)}."
                )
            return tf_other - self

    def __neg__(self):
        if self.is_encrypted:
            raw_result = shell_ops.neg_ct64(
                self._context._get_context_at_level(self._level), self._raw_tensor
            )
        else:
            raw_result = shell_ops.neg_pt64(
                self._context._get_context_at_level(self._level), self._raw_tensor
            )

        return ShellTensor64(
            _raw_tensor=raw_result,
            _context=self._context,
            _level=self._level,
            _num_mod_reductions=self._num_mod_reductions,
            _underlying_dtype=self._underlying_dtype,
            _scaling_factor=self._scaling_factor,
            _is_enc=self._is_enc,
            _is_fast_rotated=self._is_fast_rotated,
        )

    def __mul__(self, other):
        if isinstance(other, ShellTensor64):
            matched_self, matched_other = _match_moduli(self, other)

            if self.is_encrypted and other.is_encrypted:
                if self._is_fast_rotated or other._is_fast_rotated:
                    raise ValueError(
                        "A ShellTensor which has been fast-rotated or fast-reduced-summed cannot be multiplied with another ciphertext."
                    )
                raw_result = shell_ops.mul_ct_ct64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif self.is_encrypted and not other.is_encrypted:
                raw_result = shell_ops.mul_ct_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            elif not self.is_encrypted and other.is_encrypted:
                raw_result = shell_ops.mul_ct_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_other._raw_tensor,
                    matched_self._raw_tensor,
                )
            elif not self.is_encrypted and not other.is_encrypted:
                raw_result = shell_ops.mul_pt_pt64(
                    matched_self._context._get_context_at_level(matched_self._level),
                    matched_self._raw_tensor,
                    matched_other._raw_tensor,
                )
            else:
                raise ValueError("Invalid operands")

            return ShellTensor64(
                _raw_tensor=raw_result,
                _context=matched_self._context,
                _level=matched_self._level,
                _num_mod_reductions=matched_self._num_mod_reductions,
                _underlying_dtype=self._underlying_dtype,
                _scaling_factor=matched_self._scaling_factor
                * matched_other._scaling_factor,
                _is_enc=self._is_enc or other._is_enc,
                _is_fast_rotated=self._is_fast_rotated or other._is_fast_rotated,
            )
        elif isinstance(other, tf.Tensor):
            # Multiplying by a scalar uses a special op which is more efficient
            # than the caller creating a ShellTensor the same dimensions as self
            # and multiplying. Note this includes the case when broadcasting
            # over the packing dimension.
            if other.shape == () or other.shape[0] == 1:
                # If not a scalar, remove the outer dim (which is always 1) to
                # broadcast over the packing dimension.
                if other.shape != ():
                    assert other.shape[0] == 1
                    other = tf.reshape(other, other.shape[1:])

                if self.is_encrypted:
                    raw_result = shell_ops.mul_ct_tf_scalar64(
                        self._context._get_context_at_level(self._level),
                        self._raw_tensor,
                        other,
                        scaling_factor=float(self._scaling_factor),
                        random_round=_ENALBE_RANDOMIZED_ROUNDING,
                    )
                else:
                    raw_result = shell_ops.mul_pt_tf_scalar64(
                        self._context._get_context_at_level(self._level),
                        self._raw_tensor,
                        other,
                        scaling_factor=float(self._scaling_factor),
                        random_round=_ENALBE_RANDOMIZED_ROUNDING,
                    )

                return ShellTensor64(
                    _raw_tensor=raw_result,
                    _context=self._context,
                    _level=self._level,
                    _num_mod_reductions=self._num_mod_reductions,
                    _underlying_dtype=self._underlying_dtype,
                    _scaling_factor=self._scaling_factor * self._context.scaling_factor,
                    _is_enc=self._is_enc,
                    _is_fast_rotated=self._is_fast_rotated,
                )

            else:
                # Import the TensorFlow tensor to a shell plaintext and attempt
                # the multiplication again.
                shell_other = to_shell_plaintext(other, self._context)

                return self * shell_other

        else:
            # Try to import the unknown multiplicand to a TensorFlow tensor and
            # attempt the multiplication again.
            try:
                tf_other = tf.convert_to_tensor(other)
            except:
                raise ValueError(
                    f"Unsupported type for multiplication. Got {type(other)}."
                )
            return self * tf_other

    def __rmul__(self, other):
        return self * other

    def _get_generic_shell_tensor_spec(self):
        return ShellTensor64.Spec(
            _raw_tensor=tf.TensorSpec(self._raw_tensor.shape, dtype=tf.variant),
            _context=self._context._get_generic_context_spec(),
            _level=self._level,
            _num_mod_reductions=self._num_mod_reductions,
            _underlying_dtype=self._underlying_dtype,
            _scaling_factor=self._scaling_factor,
            _is_enc=self._is_enc,
            _is_fast_rotated=self._is_fast_rotated,
        )


def mod_reduce_tensor64(shell_tensor):
    """Switches the ShellTensor to a new context with different moduli. If
    preserve_plaintext is True (default), the plaintext value will be
    maintained through the modulus switch. If preserve_plaintext is False,
    the plaintext will be divided by the ratio of the new and old moduli."""

    assert isinstance(
        shell_tensor, ShellTensor64
    ), f"shell_tensor must be a ShellTensor64, instead got {type(shell_tensor)}"

    # Switch to the new context and moduli.
    if shell_tensor.is_encrypted:
        op = shell_ops.modulus_reduce_ct64
    else:
        op = shell_ops.modulus_reduce_pt64

    raw_result = op(
        shell_tensor._context._get_context_at_level(shell_tensor._level),
        shell_tensor._raw_tensor,
    )

    reduced_self = ShellTensor64(
        _raw_tensor=raw_result,
        _context=shell_tensor._context,
        _level=shell_tensor._level - 1,
        _num_mod_reductions=shell_tensor._num_mod_reductions + 1,
        _underlying_dtype=shell_tensor._underlying_dtype,
        _scaling_factor=shell_tensor._scaling_factor,
        _is_enc=shell_tensor._is_enc,
        _is_fast_rotated=shell_tensor._is_fast_rotated,
    )

    return reduced_self


def _match_moduli(x, y):
    with tf.name_scope("match_moduli"):
        # Mod switch to the smaller modulus of the two.
        while x._num_mod_reductions < y._num_mod_reductions:
            x = mod_reduce_tensor64(x)
        while x._num_mod_reductions > y._num_mod_reductions:
            y = mod_reduce_tensor64(y)

    return x, y


def _match_moduli_and_scaling(x, y):
    with tf.name_scope("match_moduli_and_scaling"):
        x, y = _match_moduli(x, y)

        # The scaling factor is a float, but if it's very close to an integer,
        # round it to avoid floating point issues.
        if abs(x._scaling_factor - round(x._scaling_factor)) < 1e-6:
            gcd = math.gcd(round(x._scaling_factor), round(y._scaling_factor))
            lcm = math.lcm(round(x._scaling_factor), round(y._scaling_factor))
        else:
            gcd = 1.0
            lcm = x._scaling_factor * y._scaling_factor

        # Match the scaling factors.
        if lcm > x._scaling_factor:
            x = x.__mul__(gcd / x._scaling_factor)
        if lcm > y._scaling_factor:
            y = y.__mul__(gcd / y._scaling_factor)

    return x, y


def largest_case_rounding(tensor, scaling_factor):
    """
    Rounds a tensor to the largest absolute fractional value of the scaling
    factor.

    This function pairs with randomized_rounding() and will return the
    largest-case randomized rounding of the tensor, i.e., the tensor with the
    largest L2 norm after being randomly rounded.
    """
    if scaling_factor == 0 or scaling_factor == float("inf"):
        return tensor

    with tf.name_scope("worst_case_rounding"):
        scaled_tensor = tensor * scaling_factor
        ceil = tf.math.ceil(scaled_tensor)
        floor = tf.math.floor(scaled_tensor)

        worst_case = tf.where(scaled_tensor > 0, ceil, floor)
        return worst_case / scaling_factor


def smallest_case_rounding(tensor, scaling_factor):
    """
    Rounds a tensor to the smallest absolute fractional value of the scaling
    factor.

    This function pairs with randomized_rounding() and will return the
    smallest-case randomized rounding of the tensor, i.e., the tensor with the
    smallest L2 norm after being randomly rounded.
    """
    if scaling_factor == 0 or scaling_factor == float("inf"):
        return tensor

    with tf.name_scope("best_case_rounding"):
        scaled_tensor = tensor * scaling_factor
        ceil = tf.math.ceil(scaled_tensor)
        floor = tf.math.floor(scaled_tensor)

        best_case = tf.where(scaled_tensor > 0, floor, ceil)
        return best_case / scaling_factor


_ENALBE_RANDOMIZED_ROUNDING = False


def enable_randomized_rounding(enable=True):
    global _ENALBE_RANDOMIZED_ROUNDING
    _ENALBE_RANDOMIZED_ROUNDING = enable


def randomized_rounding(tensor):
    """
    Performs randomized rounding on a tensorflow tensor.
    """
    with tf.name_scope("randomized_rounding"):
        ceil = tf.math.ceil(tensor)
        floor = tf.math.floor(tensor)
        decimal_part = tf.where(tensor > 0, tensor - floor, ceil - tensor)

        randomness = tf.random.uniform(tf.shape(tensor), dtype=tensor.dtype)
        round_up_mask = tf.where(
            tensor >= 0, randomness < decimal_part, randomness > decimal_part
        )

        return tf.where(round_up_mask, ceil, floor)


def _encode_scaling(tf_tensor, scaling_factor=1):
    if isinstance(tf_tensor, ShellTensor64):
        raise ValueError(f"Unsupported type for encoding. Got {type(tf_tensor)}.")

    with tf.name_scope("encode_scaling"):
        if tf_tensor.dtype in [tf.float16, tf.float32, tf.float64]:
            global _ENALBE_RANDOMIZED_ROUNDING
            if _ENALBE_RANDOMIZED_ROUNDING:
                return tf.cast(
                    randomized_rounding(tf_tensor * scaling_factor), tf.int64
                )
            else:
                return tf.cast(tf.round(tf_tensor * scaling_factor), tf.int64)
        elif tf_tensor.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            # Pass unsigned datatypes to shell as uint64.
            return tf.cast(tf_tensor, tf.uint64)
        elif tf_tensor.dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
            # Pass signed datatypes to shell as int64.
            return tf.cast(tf_tensor, tf.int64)
        else:
            raise ValueError(f"Unsupported dtype {tf_tensor.dtype}")


def _decode_scaling(scaled_tensor, output_dtype, scaling_factor):
    with tf.name_scope("decode_scaling"):
        if output_dtype in [tf.float16, tf.float32, tf.float64]:
            assert scaled_tensor.dtype == tf.int64
            return tf.cast(scaled_tensor, output_dtype) / scaling_factor
        elif output_dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            assert scaled_tensor.dtype == tf.uint64
            return tf.cast(scaled_tensor, output_dtype)
        elif output_dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
            assert scaled_tensor.dtype == tf.int64
            return tf.cast(scaled_tensor, output_dtype)
        else:
            raise ValueError(f"Unsupported dtype {output_dtype}")


def to_shell_plaintext(
    tensor, context, override_scaling_factor=None, randomized_rounding_override=None
):
    """Converts a Tensorflow tensor to a ShellTensor which holds a plaintext.
    Under the hood, this means encoding the Tensorflow tensor to a BGV style
    polynomial representation with the sign and scaling factor encoded as
    appropriate."""
    assert isinstance(
        context, ShellContext64
    ), f"Context must be a ShellContext64, instead got {type(context)}"

    if isinstance(tensor, ShellTensor64):
        if tensor.is_encrypted:
            raise ValueError(
                "Cannot use to_plaintext() on an encrypted ShellTensor. Use to_tensorflow() with a key to decrypt."
            )
        return tensor  # Do nothing, already a shell plaintext.

    elif isinstance(tensor, tf.Tensor):
        if override_scaling_factor is not None:
            scaling_factor = override_scaling_factor
        else:
            scaling_factor = context.scaling_factor

        return ShellTensor64(
            _raw_tensor=shell_ops.polynomial_import64(
                context._get_context_at_level(context.level),
                tensor,
                scaling_factor=scaling_factor,
                random_round=(
                    randomized_rounding_override
                    if randomized_rounding_override is not None
                    else _ENALBE_RANDOMIZED_ROUNDING
                ),
            ),
            _context=context,
            _level=context.level,
            _num_mod_reductions=0,
            _underlying_dtype=tensor.dtype,
            _scaling_factor=float(scaling_factor),
            _is_enc=False,
        )
    else:
        try:
            return to_shell_plaintext(tf.convert_to_tensor(tensor), context)
        except:
            raise ValueError(f"Cannot convert to ShellTensor64. Got {type(tensor)}.")


def to_encrypted(x, key, context=None):
    """Encrypts a plaintext tensor or ShellTensor using the provided key. If
    the input is a Tensorflow tensor, a context must also be provided. If the
    input is a ShellTensor, the context is ignored and the context stored
    in the input ShellTensor is used."""
    if not isinstance(key, ShellKey64):
        raise ValueError("Key must be a ShellKey64")

    if isinstance(x, ShellTensor64):
        if x._is_enc:
            return x  # Do nothing, already encrypted.
        else:
            return ShellTensor64(
                _raw_tensor=shell_ops.encrypt64(
                    x._context._get_context_at_level(x._level),
                    key._get_key_at_level(x._level),
                    x._raw_tensor,
                ),
                _context=x._context,
                _level=x._level,
                _num_mod_reductions=x._num_mod_reductions,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=True,
            )
    else:
        if not isinstance(context, ShellContext64):
            raise ValueError(
                "ShellContext64 must be provided when encrypting anything other than a ShellTensor64."
            )

        # Encode to a shell plaintext using the provided context and call
        # ourself to encrypt.
        return to_encrypted(to_shell_plaintext(x, context), key)


def to_tensorflow(s_tensor, key=None):
    """Converts a ShellTensor to a Tensorflow tensor. If the ShellTensor is
    encrypted, a key must be provided to decrypt it. If the ShellTensor is
    plaintext, the key is ignored."""
    assert isinstance(
        s_tensor, ShellTensor64
    ), f"Should be ShellTensor, instead got {type(s_tensor)}"

    # Find out what dtype shell thinks the plaintext is.
    shell_dtype = s_tensor._underlying_dtype

    try:
        batching_dim = s_tensor._context.num_slots.numpy()
    except AttributeError:
        batching_dim = -1

    if s_tensor.is_encrypted and s_tensor._is_fast_rotated:
        if not isinstance(key, ShellFastRotationKey64):
            raise ValueError(
                "ShellFastRotationKey must be provided to decrypt a fast-rotated ShellTensor."
            )

        # Get the correct rotation key for the level of this ciphertext.
        raw_rotation_key = key._get_key_at_level(s_tensor._level)

        # Decrypt op returns a tf Tensor.
        tf_tensor = shell_ops.decrypt_fast_rotated64(
            context=s_tensor._context._get_context_at_level(s_tensor._level),
            fast_rotation_key=raw_rotation_key,
            val=s_tensor._raw_tensor,
            runtime_batching_dim=s_tensor._context.num_slots,
            dtype=shell_dtype,
            batching_dim=batching_dim,
            scaling_factor=s_tensor._scaling_factor,
        )

    elif s_tensor.is_encrypted:
        if not isinstance(key, ShellKey64):
            raise ValueError(
                "Key must be provided to decrypt an encrypted ShellTensor."
            )

        # Decrypt op returns a tf Tensor.
        tf_tensor = shell_ops.decrypt64(
            context=s_tensor._context._get_context_at_level(s_tensor._level),
            key=key._get_key_at_level(s_tensor._level),
            val=s_tensor._raw_tensor,
            runtime_batching_dim=s_tensor._context.num_slots,
            dtype=shell_dtype,
            batching_dim=batching_dim,
            scaling_factor=s_tensor._scaling_factor,
        )

    elif not s_tensor.is_encrypted:
        # Convert from polynomial representation to plaintext tensorflow tensor.
        # Always convert to int64, then handle the fixed point as appropriate.
        tf_tensor = shell_ops.polynomial_export64(
            shell_context=s_tensor._context._get_context_at_level(s_tensor._level),
            val=s_tensor._raw_tensor,
            runtime_batching_dim=s_tensor._context.num_slots,
            dtype=shell_dtype,
            batching_dim=batching_dim,
            scaling_factor=s_tensor._scaling_factor,
        )

    else:
        raise ValueError(f"Invalid ShellTensor state. Got {s_tensor}.")

    return tf_tensor


def mask_with_pt(x, mask):
    if not isinstance(mask, tf.Tensor):
        raise ValueError(f"Mask must be a TensorFlow Tensor. Got {type(mask)}.")

    if isinstance(x, ShellTensor64):
        # Do not encode the mask with a scaling factor, it is 0 or 1.
        shell_mask = to_shell_plaintext(mask, x._context, override_scaling_factor=1)

        matched_x, matched_mask = _match_moduli(x, shell_mask)

        if x._is_enc:
            raw_result = shell_ops.mul_ct_pt64(
                matched_x._context._get_context_at_level(matched_x._level),
                matched_x._raw_tensor,
                matched_mask._raw_tensor,
            )
        else:
            raw_result = shell_ops.mul_pt_pt64(
                matched_x._context._get_context_at_level(matched_x._level),
                matched_x._raw_tensor,
                matched_mask._raw_tensor,
            )

        return ShellTensor64(
            _raw_tensor=raw_result,
            _context=matched_x._context,
            _level=matched_x._level,
            _num_mod_reductions=matched_x._num_mod_reductions,
            _underlying_dtype=matched_x._underlying_dtype,
            _scaling_factor=matched_x._scaling_factor,  # Same sacling factor
            _is_enc=matched_x._is_enc,
            _is_fast_rotated=matched_x._is_fast_rotated,
        )

    elif isinstance(x, tf.Tensor):
        return x * mask

    else:
        raise ValueError(f"Unsupported type for mask_with_pt. Got {type(x)}.")


def roll(x, shift, rotation_key=None):
    if isinstance(x, ShellTensor64):
        if not isinstance(rotation_key, ShellRotationKey64):
            raise ValueError(
                f"Rotation key must be provided. Instead saw {rotation_key}."
            )

        if not x._is_enc:
            raise ValueError("Unencrypted ShellTensor rotation not supported yet.")

        # Get the correct rotation key for the level of this ciphertext.
        raw_rotation_key = rotation_key._get_key_at_level(x._level)

        shift = tf.cast(shift, tf.int64)

        return ShellTensor64(
            _raw_tensor=shell_ops.roll64(
                x._context._get_context_at_level(x._level),
                raw_rotation_key,
                x._raw_tensor,
                shift,
            ),
            _context=x._context,
            _level=x._level,
            _num_mod_reductions=x._num_mod_reductions,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=True,
            _is_fast_rotated=x._is_fast_rotated,
        )
    elif isinstance(x, tf.Tensor):
        # TensorFlow's roll has slightly different semantics than hadal's
        # roll. Encrypted rotation affects top and bottom halves independently.
        # This function emulates this in plaintext by splitting the tensor in
        # half, rotating each half, and then concatenating them back together.
        top, bottom = tf.split(x, num_or_size_splits=2, axis=0)
        top = tf.roll(top, shift, axis=0)
        bottom = tf.roll(bottom, shift, axis=0)
        rotated_tftensor = tf.concat([top, bottom], axis=0)
        return rotated_tftensor

    else:
        raise ValueError(f"Unsupported type for roll. Got {type(x)}.")


def reduce_sum(x, axis, rotation_key=None):
    if isinstance(x, ShellTensor64):
        if not x._is_enc:
            raise ValueError("Unencrypted ShellTensor reduce_sum not supported yet.")

        # Check axis is a scalar
        if isinstance(axis, tf.Tensor) and not axis.shape != []:
            raise ValueError("Only scalar `axis` is supported.")

        if axis == 0:
            if not isinstance(rotation_key, ShellRotationKey64):
                raise ValueError(
                    f"Rotation key must be provided to reduce_sum over the first axis. Instead saw {rotation_key}."
                )

            # Get the correct rotation key for the level of this ciphertext.
            raw_rotation_key = rotation_key._get_key_at_level(x._level)

            return ShellTensor64(
                _raw_tensor=shell_ops.reduce_sum_by_rotation_ct64(
                    x._context._get_context_at_level(x._level),
                    raw_rotation_key,
                    x._raw_tensor,
                ),
                _context=x._context,
                _level=x._level,
                _num_mod_reductions=x._num_mod_reductions,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=True,
                _is_fast_rotated=x._is_fast_rotated,
            )

        else:
            return ShellTensor64(
                _raw_tensor=shell_ops.reduce_sum_ct64(
                    x._context._get_context_at_level(x._level),
                    x._raw_tensor,
                    axis=axis,
                    reduce_dim_size=x.shape[axis] if x.shape[axis] is not None else -1,
                ),
                _context=x._context,
                _level=x._level,
                _num_mod_reductions=x._num_mod_reductions,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=True,
                _is_fast_rotated=x._is_fast_rotated,
            )
    elif isinstance(x, tf.Tensor):
        if axis == 0:
            # TensorFlow's reduce_sum over axis 0 (the slotting dimension) has
            # slightly different semantics than hadal's reduce_sum. Encrypted
            # reduce_sum affects top and bottom halves independently, as well as
            # repeating the sum across the halves. This emulates this in
            # plaintext.
            half_slots = x.shape[0] // 2
            bottom_answer = tf.math.reduce_sum(x[0:half_slots], axis=0, keepdims=True)
            top_answer = tf.math.reduce_sum(x[half_slots:], axis=0, keepdims=True)

            repeated_bottom_answer = tf.repeat(
                bottom_answer, repeats=half_slots, axis=0
            )
            repeated_top_answer = tf.repeat(top_answer, repeats=half_slots, axis=0)

            return tf.concat([repeated_bottom_answer, repeated_top_answer], 0)
        else:
            return tf.reduce_sum(x, axis)
    else:
        raise ValueError(f"Unsupported type for reduce_sum. Got {type(x)}.")


def reduce_sum_with_mod(x, axis, context, scaling_factor=None):
    if not isinstance(x, tf.Tensor):
        raise ValueError(f"Input must be a TensorFlow tensor. Got {type(x)}.")

    if not isinstance(context, ShellContext64):
        raise ValueError(f"Context must be a ShellContext64. Got {type(context)}.")

    if scaling_factor == None:
        s = context.scaling_factor
    else:
        s = scaling_factor

    # Shell tensor represents floats as integers * scaling_factor.
    scaled_x = _encode_scaling(x, s)

    # The context is only used to get the plaintext modulus, any level will do.
    c = context._get_context_at_level(1)

    reduced_x = shell_ops.reduce_sum_with_modulus_pt64(c, scaled_x, axis=axis)
    return _decode_scaling(reduced_x, x.dtype, s)


def fast_reduce_sum(x):
    """Fast reduce sum is a special case of reduce sum where the encrypted
    input is reduce_summed, however it skips the keyswitching so the resulting
    ciphertext is no longer valid under the original secret key. The plaintext
    can still be recovered, through a special decryption process. This means
    a fast_reduce_summed ciphertext has limited subsequent operations, i.e. only
    add / multiply by plaintexts are supported. See the op kernel for a more
    technical explanation."""
    if not isinstance(x, ShellTensor64):
        raise ValueError("Input must be ShellTensor.")
    if not x._is_enc:
        raise ValueError("Unencrypted fast_reduce_sum not supported yet.")
    if x._is_fast_rotated:
        raise ValueError("Cannot fast_reduce_sum a fast_rotated ShellTensor.")

    return ShellTensor64(
        _raw_tensor=shell_ops.fast_reduce_sum_by_rotation64(
            x._context._get_context_at_level(x._level), x._raw_tensor
        ),
        _context=x._context,
        _level=x._level,
        _num_mod_reductions=x._num_mod_reductions,
        _underlying_dtype=x._underlying_dtype,
        _scaling_factor=x._scaling_factor,
        _is_enc=True,
        _is_fast_rotated=True,
    )


def matmul(x, y, rotation_key=None, pt_ct_reduction="galois", emulate_pt_ct=False):
    """Matrix multiplication is specialized to whether the operands are
    plaintext or ciphertext.

    matmul(ciphertext, plaintext) works the same way as Tensorflow.

    matmul(plaintext, ciphertext) in hadal has slightly different semantics
    than plaintext / Tensorflow. hadal affects top and bottom halves
    independently, as well as the first dimension repeating the sum of either
    the halves."""

    if len(x.shape) < 2 or len(y.shape) < 2:
        raise ValueError(
            f"matmul not supported for tensors with rank < 2. Got {x.shape} and {y.shape}."
        )

    if isinstance(x, ShellTensor64) and isinstance(y, tf.Tensor):
        if x._underlying_dtype != y.dtype:
            raise ValueError(
                f"Underlying dtypes must match. Got {x._underlying_dtype} and {y.dtype}"
            )

        # Encode the plaintext y to the same scaling factor as x.
        scaled_y = _encode_scaling(y, x._scaling_factor)

        return ShellTensor64(
            _raw_tensor=shell_ops.mat_mul_ct_pt64(
                x._context._get_context_at_level(x._level),
                x._raw_tensor,
                scaled_y,
                reduce_dim_size=x.shape[-1],
            ),
            _context=x._context,
            _level=x._level,
            _num_mod_reductions=x._num_mod_reductions,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor**2,
            _is_enc=True,
            _is_fast_rotated=x._is_fast_rotated,
        )

    elif isinstance(x, tf.Tensor) and isinstance(y, ShellTensor64):
        if x.dtype != y._underlying_dtype:
            raise ValueError(
                f"Underlying dtypes must match. Got {x.dtype} and {y._underlying_dtype}"
            )

        if pt_ct_reduction not in ["galois", "fast", "none"]:
            raise ValueError(
                f"pt_ct_reduction must be 'galois', 'fast', or 'none'. Got {pt_ct_reduction}."
            )

        # Encode the plaintext x to the same scaling factor as y.
        scaled_x = _encode_scaling(x, y._context.scaling_factor)

        if pt_ct_reduction == "galois":
            if not isinstance(rotation_key, ShellRotationKey64):
                raise ValueError(
                    f"Rotation key must be provided to matmul pt*ct with galois reduction. Instead saw {rotation_key}."
                )
            # Get the correct rotation key for the level of y.
            raw_rotation_key = rotation_key._get_key_at_level(y._level)
        elif pt_ct_reduction == "fast":
            if y._is_fast_rotated:
                raise ValueError(
                    "A ShellTensor which has been fast-reduced-summed cannot be fast-reduced-summed again."
                )
            # Any variant tensor will do. It is ignored by the op.
            raw_rotation_key = y._context._get_context_at_level(y._level)
        elif pt_ct_reduction == "none":
            # Any variant tensor will do. It is ignored by the op.
            raw_rotation_key = y._context._get_context_at_level(y._level)

        return ShellTensor64(
            _raw_tensor=shell_ops.mat_mul_pt_ct64(
                y._context._get_context_at_level(y._level),
                scaled_x,
                y._raw_tensor,
                raw_rotation_key,
                reduction=pt_ct_reduction,
            ),
            _context=y._context,
            _level=y._level,
            _num_mod_reductions=y._num_mod_reductions,
            _underlying_dtype=y._underlying_dtype,
            _scaling_factor=y._scaling_factor * y._context.scaling_factor,
            _is_enc=True,
            _is_fast_rotated=pt_ct_reduction == "fast",
        )

    elif isinstance(x, ShellTensor64) and isinstance(y, ShellTensor64):
        return NotImplementedError

    elif isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
        if emulate_pt_ct:
            # hadal matmult has slightly different semantics than plaintext /
            # Tensorflow. Encrypted matmult affects top and bottom halves
            # independently, as well as the first dimension repeating the sum of
            # either the halves. This function emulates this in plaintext with
            # element-wise multiplication, and an optional reduction.
            shape_range = range(len(x.shape))
            x = tf.transpose(x, perm=[shape_range[-1]] + list(shape_range[:-1]))
            x = tf.expand_dims(x, axis=-1)
            for _ in range(len(x.shape) - 2):
                y = tf.expand_dims(y, axis=-2)
            res = x * y

            if pt_ct_reduction != "none":
                res = reduce_sum(res, axis=0)

            return res

        else:
            return tf.matmul(x, y)

    else:
        raise ValueError(
            f"Unsupported types for matmul. Got {type(x)} and {type(y)}. If multiplying by a plaintext, pass it as a plain TensorFlow tensor, not a ShellTensor."
        )


def expand_dims(x, axis=-1):
    if isinstance(x, ShellTensor64):
        # Perform some checks on the axis.
        if axis == 0:
            raise ValueError(
                "Cannot expand dims at axis 0 for ShellTensor64, this is the batching dimension."
            )
        return ShellTensor64(
            _raw_tensor=shell_ops.expand_dims_variant(x._raw_tensor, axis=axis),
            _context=x._context,
            _level=x._level,
            _num_mod_reductions=x._num_mod_reductions,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=x._is_enc,
            _is_fast_rotated=x._is_fast_rotated,
        )
    elif isinstance(x, tf.Tensor):
        return tf.expand_dims(x, axis)
    else:
        raise ValueError("Unsupported type for expand_dims")


def reshape(x, shape):
    if isinstance(x, ShellTensor64):
        # Perform some checks on the new shape.
        if tf.executing_eagerly() and shape[0] != x._context.num_slots:
            raise ValueError(
                "Cannot reshape axis 0 for ShellTensor64, this is the batching dimension."
            )
        return ShellTensor64(
            _raw_tensor=tf.reshape(x._raw_tensor, shape[1:]),
            _context=x._context,
            _level=x._level,
            _num_mod_reductions=x._num_mod_reductions,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=x._is_enc,
            _is_fast_rotated=x._is_fast_rotated,
        )
    elif isinstance(x, tf.Tensor):
        return tf.reshape(x, shape)
    else:
        raise ValueError("Unsupported type for reshape")


def shape(x):
    if isinstance(x, ShellTensor64):
        return tf.concat(
            [
                tf.expand_dims(tf.cast(x._context.num_slots, dtype=tf.int32), axis=0),
                tf.shape(x._raw_tensor),
            ],
            axis=0,
        )
    elif isinstance(x, tf.Tensor):
        return tf.shape(x)
    else:
        raise ValueError("Unsupported type for shape")


def broadcast_to(x, shape):
    if isinstance(x, ShellTensor64):
        if (
            tf.executing_eagerly()
            and tf.cast(shape[0], tf.int64) != x._context.num_slots
        ):
            raise ValueError(
                "Cannot broadcast_to over axis 0 for ShellTensor64, this is the batching dimension."
            )

        return ShellTensor64(
            _raw_tensor=tf.broadcast_to(x._raw_tensor, shape[1:]),
            _context=x._context,
            _level=x._level,
            _num_mod_reductions=x._num_mod_reductions,
            _underlying_dtype=x._underlying_dtype,
            _scaling_factor=x._scaling_factor,
            _is_enc=x._is_enc,
            _is_fast_rotated=x._is_fast_rotated,
        )
    elif isinstance(x, tf.Tensor):
        return tf.broadcast_to(x, shape)
    else:
        raise ValueError("Unsupported type for broadcast_to")


def split(x, num_or_size_splits, axis=0, num=None):
    if isinstance(x, ShellTensor64):
        if axis == 0:
            raise ValueError(
                "Cannot split over axis 0 for ShellTensor64, this is the batching dimension."
            )

        split_raw_tensors = tf.split(x._raw_tensor, num_or_size_splits, axis - 1, num)
        return [
            ShellTensor64(
                _raw_tensor=r,
                _context=x._context,
                _level=x._level,
                _num_mod_reductions=x._num_mod_reductions,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=x._is_enc,
                _is_fast_rotated=x._is_fast_rotated,
            )
            for r in split_raw_tensors
        ]
    elif isinstance(x, tf.Tensor):
        return tf.split(x, num_or_size_splits, axis, num)
    else:
        raise ValueError("Unsupported type for split")


def all_but_data_equal(x):
    return tf.reduce_all(
        tf.concat(
            [
                [
                    tf.equal(
                        xi._context.plaintext_modulus, x[0]._context.plaintext_modulus
                    )
                    for xi in x
                ],
                tf.reshape(
                    [
                        tf.equal(xi._context.main_moduli, x[0]._context.main_moduli)
                        for xi in x
                    ],
                    [-1],
                ),
                [tf.equal(xi._level, x[0]._level) for xi in x],
                [
                    tf.equal(xi._num_mod_reductions, x[0]._num_mod_reductions)
                    for xi in x
                ],
                [xi._underlying_dtype == x[0]._underlying_dtype for xi in x],
                [tf.equal(xi._scaling_factor, x[0]._scaling_factor) for xi in x],
                [tf.equal(xi._is_enc, x[0]._is_enc) for xi in x],
                [tf.equal(xi._is_fast_rotated, x[0]._is_fast_rotated) for xi in x],
            ],
            axis=0,
        )
    )


def concat(x, axis=0):
    if all([isinstance(xi, ShellTensor64) for xi in x]):
        if axis == 0:
            raise ValueError(
                "Cannot concat over axis 0 for ShellTensor64, this is the batching dimension."
            )

        if len(x) == 0:
            raise ValueError("Cannot concat an empty list of ShellTensor64.")

        if len(x) == 1:
            return x[0]

        # All tensors must match.
        tf.debugging.assert_equal(
            all_but_data_equal(x),
            True,
            message="All tensors must have the same properties.",
        )

        if x[0]._is_enc:
            concat_raw_tensor = shell_ops.concat_ct64(
                axis=axis - 1, values=[xi._raw_tensor for xi in x]
            )
        else:
            concat_raw_tensor = shell_ops.concat_pt64(
                axis=axis - 1, values=[xi._raw_tensor for xi in x]
            )

        return ShellTensor64(
            _raw_tensor=concat_raw_tensor,
            _context=x[0]._context,
            _level=x[0]._level,
            _num_mod_reductions=x[0]._num_mod_reductions,
            _underlying_dtype=x[0]._underlying_dtype,
            _scaling_factor=x[0]._scaling_factor,
            _is_enc=x[0]._is_enc,
            _is_fast_rotated=x[0]._is_fast_rotated,
        )

    elif all([isinstance(xi, tf.Tensor) for xi in x]):
        return tf.concat(x, axis)
    else:
        raise ValueError("Unsupported type for concat")


def segment_sum(
    x,
    segments,
    num_segments,
    rotation_key=None,
    reduction="galois",
    skip_pt_counts=False,
):
    if not isinstance(segments, tf.Tensor):
        raise ValueError("`segments` must be a TensorFlow tensor.")

    if isinstance(x, ShellTensor64):
        if reduction not in ["galois", "none"]:
            raise ValueError(f"Reduction must be 'galois' or 'none'. Got {reduction}.")

        if reduction == "galois":
            if not isinstance(rotation_key, ShellRotationKey64):
                raise ValueError(
                    f"Rotation key must be provided for galois-based reduction. Instead saw {rotation_key}."
                )
            raw_rotation_key = rotation_key._get_key_at_level(x._level)

        elif reduction == "none":
            # Any variant tensor will do. It is ignored by the op.
            raw_rotation_key = x._context._get_context_at_level(x._level)

        raw_result, reduction_count = shell_ops.segment_sum_ct(
            x._context._get_context_at_level(x._level),
            x._raw_tensor,
            segments,
            num_segments,
            raw_rotation_key,
            reduction=reduction,
        )

        return (
            ShellTensor64(
                _raw_tensor=raw_result,
                _context=x._context,
                _level=x._level,
                _num_mod_reductions=x._num_mod_reductions,
                _underlying_dtype=x._underlying_dtype,
                _scaling_factor=x._scaling_factor,
                _is_enc=x._is_enc,
                _is_fast_rotated=x._is_fast_rotated,
            ),
            reduction_count,
        )
    elif isinstance(x, tf.Tensor):
        # hadal segment functions differs from tensorflow in the
        # following ways:
        # 1) The ciphertext dimension is included in the output, but only one
        # dimension is valid. For the top half of the ciphertext, the first
        # dimension is valid, and for the bottom half, the `num_slots // 2`th
        # dimension is valid.
        # 2) The reduction (if enabled) only happens across half of the batching
        # dimension, due to how rotations in hadal work. Segment reduction
        # happens on the top and bottom halves of the ciphertext independently.

        # TensorFlow does not have a GPU implementation of bincount. The caller
        # can skip the bincount computation if the output is not necessary and
        # using a GPU.
        if skip_pt_counts:
            pt_counts = None
        else:

            def bincount(x):
                return tf.math.bincount(
                    x,
                    minlength=num_segments,
                    maxlength=num_segments,
                    dtype=segments.dtype,
                )

            segments_nonnegative = tf.where(segments >= 0, segments, num_segments + 1)
            pt_counts = tf.map_fn(
                bincount, segments_nonnegative, fn_output_signature=segments.dtype
            )

        if reduction == "none":

            # TensorFlow's segment_sum does not work "per example" or over the
            # first dimentsion where hadal does. Emulate this with map_fn.
            def per_example_segment_sum(tupl):
                x, segments = tupl

                # Fake batch size.
                x = tf.expand_dims(x, 0)
                segments = tf.expand_dims(segments, 0)

                return tf.math.unsorted_segment_sum(x, segments, num_segments)

            output_signature = tf.TensorSpec.from_tensor(
                per_example_segment_sum((x[0], segments[0]))
            )
            res = tf.map_fn(
                per_example_segment_sum,
                (x, segments),
                fn_output_signature=output_signature,
                name="shell_segment_sum_tf",
            )
            return res, pt_counts

        else:
            x_top, x_bottom = tf.split(x, num_or_size_splits=2, axis=0)

            padding = tf.zeros_like(x_top)

            x_top = tf.concat([x_top, padding], 0)
            x_bottom = tf.concat([padding, x_bottom], 0)

            res_top = tf.math.unsorted_segment_sum(x_top, segments, num_segments)
            res_bottom = tf.math.unsorted_segment_sum(x_bottom, segments, num_segments)

            res_top = tf.expand_dims(res_top, 0)
            res_top = tf.repeat(res_top, repeats=x.shape[0] // 2, axis=0)
            res_top = tf.concat([res_top, tf.zeros_like(res_top)], 0)

            res_bottom = tf.expand_dims(res_bottom, 0)
            res_bottom = tf.repeat(res_bottom, repeats=x.shape[0] // 2, axis=0)
            res_bottom = tf.concat([tf.zeros_like(res_bottom), res_bottom], 0)

            res_top = tf.expand_dims(res_top, 1)
            res_bottom = tf.expand_dims(res_bottom, 1)
            res = tf.concat([res_top, res_bottom], 1)
            return res, pt_counts

    else:
        raise ValueError("Unsupported type for segment_sum")


def _conv2d(x, filt, strides, padding, dilations, output_shape, func):
    if not x._is_enc and not filt._is_enc:
        raise ValueError("At least one input must be encrypted ShellTensor64.")

    if x._is_fast_rotated or filt._is_fast_rotated:
        raise ValueError(
            "A ShellTensor which has been fast-rotated or fast-reduced-summed cannot be an input to conv2d."
        )

    matched_x, matched_filt = _match_moduli(x, filt)

    if output_shape is not None:
        for i, dim in enumerate(output_shape):
            if dim is None:
                output_shape[i] = -1

    return ShellTensor64(
        _raw_tensor=func(
            matched_x._context._get_context_at_level(matched_x._level),
            matched_x._raw_tensor,
            matched_filt._raw_tensor,
            strides,
            padding,
            dilations,
            filter_num_elements=matched_filt._raw_tensor.shape.num_elements(),
            output_shape=output_shape,
        ),
        _context=matched_x._context,
        _level=matched_x._level,
        _num_mod_reductions=matched_x._num_mod_reductions,
        _underlying_dtype=matched_x._underlying_dtype,
        _scaling_factor=matched_x._scaling_factor * matched_filt._scaling_factor,
        _is_enc=True,
        _is_fast_rotated=False,
    )


def conv2d(
    x,
    filt,
    strides=[1, 1, 1, 1],
    padding=[0, 0, 0, 0],
    dilations=[1, 1, 1, 1],
    with_channel=False,
    output_shape=None,
):
    """Convolution (technically cross-correlation) of x with filt.

    This operation is different from TensorFlow's conv2d in that it deconvolves
    a batch of filters, not a single filter.

    x and filt can be ShellTensors or TensorFlow tensors. If both are TensorFlow
    tensors, the output is a TensorFLow tensor which mimics the behavior of the
    hadal operation.

    Important note: Tensorflow (and this function) flip the order of the filter
    argument between conv2d and conv2d_transpose. conv2d expects the filter to
    be of shape [filter_height, filter_width, in_channels, out_channels].

    x is expected to be of shape:
    [batch, in_height, in_width, in_channels].

    The order of strides padding, and dilations is top, bottom, left, right.
    """

    # Plaintext implementation of hadal's conv2d using tensorflow ops.
    if not isinstance(x, ShellTensor64) and not isinstance(filt, ShellTensor64):
        # When the number of channels in x and filt are the same, perform
        # element-wise convolution for each x and filt pair in the batch.
        if not with_channel:
            tf_padding = [
                [0, 0],
                [padding[0], padding[1]],  # top, bottom
                [padding[2], padding[3]],  # left, right
                [0, 0],
            ]

            def single_conv(tupl):
                x, kernel = tupl
                x = tf.expand_dims(x, 0)  # Fake batch size.
                return tf.nn.conv2d(
                    x, kernel, strides=strides, padding=tf_padding, dilations=dilations
                )

            output_signature = tf.TensorSpec.from_tensor(single_conv((x[0], filt[0])))
            res = tf.map_fn(
                single_conv,
                (x, filt),
                fn_output_signature=output_signature,
                name="shell_conv2dtf",
            )
            res = tf.squeeze(res, axis=1)
        else:
            # When the number of channels in x and filt are different, mimic
            # hadal's behavior and slide over the channels dimension.
            #
            # Manually pad the input tensor to allow explicit padding.
            x = tf.pad(
                x, [[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]]
            )

            # TensorFlow expects dilations in the old channel dimension.
            exp_dilations = dilations + [1]
            exp_dilations[-2] = exp_dilations[-3]

            def single_conv(tupl):
                x, kernel = tupl
                x = tf.expand_dims(x, 0)  # Fake batch size.
                return tf.nn.conv3d(
                    x,
                    kernel,
                    strides=strides + [1],
                    padding="VALID",
                    dilations=exp_dilations,
                )

            # Use a 3d convolution with dummy channel dimension.
            x_exp = tf.expand_dims(x, -1)
            filt_exp = tf.expand_dims(filt, -2)
            output_signature = tf.TensorSpec.from_tensor(
                single_conv((x_exp[0], filt_exp[0]))
            )
            res = tf.map_fn(
                single_conv,
                (x_exp, filt_exp),
                fn_output_signature=output_signature,
                name="shell_conv2dtf_with_chan",
            )
            res = tf.squeeze(res, axis=1)  # Remove fake batch size.

        # Trim the output shape to the requested size. When using a
        # convolutional layer, the output shape on the backward pass will always
        # be the same or smaller than the computed output shape, so trimming by
        # slicing is okay (vs. needing to pad).
        if output_shape is not None:
            res = tf.slice(res, [0, 0, 0, 0, 0], output_shape)
        return res

    if not isinstance(x, ShellTensor64):
        x = to_shell_plaintext(x, filt._context)
    if not isinstance(filt, ShellTensor64):
        filt = to_shell_plaintext(filt, x._context)

    if not with_channel:
        # If the number of channels is equal, use the version of the op which
        # assumes x and filt have the same number of channels. This is the same
        # way TensorFlow's conv2d works.
        if x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_ct_ct64
        elif x._is_enc and not filt._is_enc:
            func = shell_ops.conv2d_ct_pt64
        elif not x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_pt_ct64
    else:
        # If the number of channels is different, use the version of the op
        # which slides over the channels dimension. This is unique to hadal.
        # TODO: When hadal has a squeeze op, all cases can use the
        # "with chan" version of the op and squeeze the channel dimension out
        # of the result.
        if x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_with_chan_ct_ct64
        elif x._is_enc and not filt._is_enc:
            func = shell_ops.conv2d_with_chan_ct_pt64
        elif not x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_with_chan_pt_ct64

    return _conv2d(x, filt, strides, padding, dilations, output_shape, func)


def conv2d_transpose(
    x,
    filt,
    strides=[1, 1, 1, 1],
    padding=[0, 0, 0, 0],
    with_channel=False,
    output_shape=None,
):
    """Deconvolution (or gradient of conv2d) of x with filt.

    This operation is different from TensorFlow's conv2d in that it
    deconvolves a batch of filters, not a single filter.

    x and filt can be ShellTensors or TensorFlow tensors. If both are TensorFlow
    tensors, the output is a TensorFLow tensor which mimics the behavior of the
    hadal operation.

    Important note: Tensorflow (and this function) flip the order of the filter
    argument between conv2d and conv2d_transpose. conv2d_transpose expects the
    filter to be of shape:
    [filter_height, filter_width, out_channels, in_channels].

    x is expected to be of shape:
    [batch, in_height, in_width, in_channels].

    The order of strides and padding is top, bottom, left, right.
    """

    # Plaintext implementation of hadal's conv2d using tensorflow ops.
    if not isinstance(x, ShellTensor64) and not isinstance(filt, ShellTensor64):
        # When the number of channels in x and filt are the same, perform
        # element-wise convolution for each x and filt pair in the batch.
        if not with_channel:
            tf_padding = [
                [0, 0],
                [padding[0], padding[1]],  # top, bottom
                [padding[2], padding[3]],  # left, right
                [0, 0],
            ]

            # If an output shape is provided, set the batching dim to 1.
            tf_output_shape = output_shape
            if tf_output_shape is not None:
                tf_output_shape[0] = 1
            else:
                tf_output_shape = [
                    1,  # Fake batch size.
                    ((x.shape[1] - 1) * strides[1])
                    + filt.shape[1]
                    - padding[0]
                    - padding[1],
                    ((x.shape[2] - 1) * strides[2])
                    + filt.shape[2]
                    - padding[2]
                    - padding[3],
                    filt.shape[3],  # Output channel dim.
                ]

            def single_conv(tupl):
                x, kernel = tupl
                x = tf.expand_dims(x, 0)  # Fake batch dimension.
                return tf.nn.conv2d_transpose(
                    x, kernel, tf_output_shape, strides=strides, padding=tf_padding
                )

            output_signature = tf.TensorSpec.from_tensor(single_conv((x[0], filt[0])))
            res = tf.map_fn(
                single_conv, (x, filt), fn_output_signature=output_signature
            )
            res = tf.squeeze(res, axis=1)  # Remove fake batch size.
        else:
            # When the number of channels in x and filt are different, mimic
            # hadal's behavior and slide over the channels dimension.
            if padding != [0, 0, 0, 0]:
                raise ValueError(
                    "Padding is not supported for plaintext conv2d_transpose when the number of channels in x and filt are different."
                )

            # If an output shape is provided, set the batching dim to 1.
            tf_output_shape = output_shape
            if tf_output_shape is not None:
                tf_output_shape[0] = 1
            else:
                tf_output_shape = [
                    1,  # Fake batch size.
                    ((x.shape[1] - 1) * strides[1])
                    + filt.shape[1]
                    - padding[0]
                    - padding[1],
                    ((x.shape[2] - 1) * strides[2])
                    + filt.shape[2]
                    - padding[2]
                    - padding[3],
                    ((filt.shape[3] - 1) * strides[3])
                    + filt.shape[3],  # Conv channel dim.
                    filt.shape[4],  # Output channel dim.
                ]

            def single_conv(tupl):
                x, kernel = tupl
                x = tf.expand_dims(x, 0)  # Fake batch size.
                return tf.nn.conv3d_transpose(
                    x, kernel, tf_output_shape, strides=strides + [1], padding="VALID"
                )

            # Use a 3d convolution with dummy channel dimension.
            x_exp = tf.expand_dims(x, -1)
            filt_exp = tf.expand_dims(filt, -2)
            output_signature = tf.TensorSpec.from_tensor(
                single_conv((x_exp[0], filt_exp[0]))
            )
            res = tf.map_fn(
                single_conv, (x_exp, filt_exp), fn_output_signature=output_signature
            )
            res = tf.squeeze(res, axis=1)  # Remove fake batch size.

        return res

    if not isinstance(x, ShellTensor64):
        x = to_shell_plaintext(x, filt._context)
    if not isinstance(filt, ShellTensor64):
        filt = to_shell_plaintext(filt, x._context)

    if not with_channel:
        # If the number of channels is equal, use the version of the op which
        # assumes x and filt have the same number of channels. This is the same
        # way TensorFlow's conv2d works.
        if x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_transpose_ct_ct64
        elif x._is_enc and not filt._is_enc:
            func = shell_ops.conv2d_transpose_ct_pt64
        elif not x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_transpose_pt_ct64
    else:
        # If the number of channels is different, use the version of the op
        # which slides over the channels dimension. This is unique to hadal.
        # TODO: When hadal has a squeeze op, all cases can use the
        # "with chan" version of the op and squeeze the channel dimension out
        # of the result.
        if x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_transpose_with_chan_ct_ct64
        elif x._is_enc and not filt._is_enc:
            func = shell_ops.conv2d_transpose_with_chan_ct_pt64
        elif not x._is_enc and filt._is_enc:
            func = shell_ops.conv2d_transpose_with_chan_pt_ct64

    return _conv2d(x, filt, strides, padding, [1, 1, 1, 1], output_shape, func)


def max_unpool2d(
    updates,
    argmax,
    pool_size=[1, 1],
    strides=[1, 1, 1, 1],
    padding="VALID",
    output_shape=None,
):
    """Max pool 2d transpose of updates with argmax. This computes the gradient of
    tf.nn.max_pool_with_argmax.

    updates is expected to be of shape:
    [batch, in_height, in_width, in_channels].

    argmax is expected to be of shape:
    [batch * out_height * out_width * in_channels].

    The output is of shape:
    [batch, out_height, out_width, in_channels].
    """
    if isinstance(updates, tf.Tensor):
        # TensorFlow does not have an unpool op. This implementation is borrowed
        # from tensorflow_addons max_unpooling_2d.
        #
        # This function currently does not support outputs of MaxPoolingWithArgMax in following cases:
        # - include_batch_in_index equals true.
        # - The max pooling operation results in duplicate values in updates and mask.
        #
        # Unpool the outputs of a maximum pooling operation.
        mask = tf.cast(argmax, "int32")
        input_shape = tf.shape(updates, out_type="int32")
        input_shape = [updates.shape[i] or input_shape[i] for i in range(4)]

        # Calculates indices for batch, height, width and feature maps.
        one_like_mask = tf.ones_like(mask, dtype="int32")
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype="int32"), shape=batch_shape
        )
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype="int32")
        f = one_like_mask * feature_range

        # Transposes indices & reshape update values to one dimension.
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    if updates._is_fast_rotated:
        raise ValueError(
            "A ShellTensor which has been fast-rotated or fast-reduced-summed cannot be an input to max_pool2d_transpose."
        )

    if padding.upper() == "VALID":
        padding_list = [0, 0, 0, 0]
    elif padding.upper() == "SAME":

        def compute_padding_same(input_size, kernel_size, stride=1, dilation_rate=1):
            import math

            dilated_kernel_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
            output_size = math.ceil(input_size / stride)
            total_padding = (
                (output_size - 1) * stride + dilated_kernel_size - input_size
            )
            padding_before = total_padding // 2
            padding_after = total_padding - padding_before
            return padding_before, padding_after

        top, bottom = compute_padding_same(updates.shape[1], pool_size[0], strides[0])
        left, right = compute_padding_same(updates.shape[2], pool_size[1], strides[1])
        padding_list = [top, bottom, left, right]
    else:
        raise ValueError(f"Padding must be 'VALID' or 'SAME'. Got {padding}.")

    return ShellTensor64(
        _raw_tensor=shell_ops.max_unpool2d_ct64(
            updates._context._get_context_at_level(updates._level),
            updates._raw_tensor,
            argmax,
            pool_size=pool_size,
            strides=strides,
            padding=padding_list,
            output_shape=output_shape,
        ),
        _context=updates._context,
        _level=updates._level,
        _num_mod_reductions=updates._num_mod_reductions,
        _underlying_dtype=updates._underlying_dtype,
        _scaling_factor=updates._scaling_factor,
        _is_enc=True,
        _is_fast_rotated=False,
    )
