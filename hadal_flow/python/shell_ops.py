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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

shell_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_shell_ops.so")
)
context_import64 = shell_ops.context_import64
auto_shell_context64 = shell_ops.auto_shell_context64
polynomial_import64 = shell_ops.polynomial_import64
polynomial_export64 = shell_ops.polynomial_export64
key_gen64 = shell_ops.key_gen64

encrypt64 = shell_ops.encrypt64
decrypt64 = shell_ops.decrypt64

# Add and subtract.
add_ct_ct64 = shell_ops.add_ct_ct64
add_ct_pt64 = shell_ops.add_ct_pt64
add_pt_pt64 = shell_ops.add_pt_pt64
sub_ct_ct64 = shell_ops.sub_ct_ct64
sub_ct_pt64 = shell_ops.sub_ct_pt64
sub_pt_pt64 = shell_ops.sub_pt_pt64
neg_ct64 = shell_ops.neg_ct64
neg_pt64 = shell_ops.neg_pt64

# Multiply.
mul_ct_ct64 = shell_ops.mul_ct_ct64
mul_ct_pt64 = shell_ops.mul_ct_pt64
mul_ct_tf_scalar64 = shell_ops.mul_ct_tf_scalar64
mul_pt_tf_scalar64 = shell_ops.mul_pt_tf_scalar64
mul_pt_pt64 = shell_ops.mul_pt_pt64
mat_mul_ct_pt64 = shell_ops.mat_mul_ct_pt64
mat_mul_pt_ct64 = shell_ops.mat_mul_pt_ct64

# Rotate slots.
rotation_key_gen64 = shell_ops.rotation_key_gen64
roll64 = shell_ops.roll64
reduce_sum_by_rotation_ct64 = shell_ops.reduce_sum_by_rotation_ct64
reduce_sum_ct64 = shell_ops.reduce_sum_ct64
reduce_sum_with_modulus_pt64 = shell_ops.reduce_sum_with_modulus_pt64

# Fast rotations
fast_rotation_key_gen64 = shell_ops.fast_rotation_key_gen64
fast_reduce_sum_by_rotation64 = shell_ops.fast_reduce_sum_by_rotation64
decrypt_fast_rotated64 = shell_ops.decrypt_fast_rotated64

# Modulus switching.
modulus_reduce_context64 = shell_ops.modulus_reduce_context64
modulus_reduce_key64 = shell_ops.modulus_reduce_key64
modulus_reduce_ct64 = shell_ops.modulus_reduce_ct64
modulus_reduce_pt64 = shell_ops.modulus_reduce_pt64

# Shape manipulation.
expand_dims_variant = shell_ops.expand_dims_variant
concat_ct64 = shell_ops.concat_ct64
concat_pt64 = shell_ops.concat_pt64

# Segmenmt-based reductions.
segment_sum_ct = shell_ops.unsorted_ct_segment_sum

# Convolution.
conv2d_pt_ct64 = shell_ops.conv2d_pt_ct64
conv2d_ct_pt64 = shell_ops.conv2d_ct_pt64
conv2d_ct_ct64 = shell_ops.conv2d_ct_ct64
conv2d_with_chan_pt_ct64 = shell_ops.conv2d_with_chan_pt_ct64
conv2d_with_chan_ct_pt64 = shell_ops.conv2d_with_chan_ct_pt64
conv2d_with_chan_ct_ct64 = shell_ops.conv2d_with_chan_ct_ct64
conv2d_transpose_pt_ct64 = shell_ops.conv2d_transpose_pt_ct64
conv2d_transpose_ct_pt64 = shell_ops.conv2d_transpose_ct_pt64
conv2d_transpose_ct_ct64 = shell_ops.conv2d_transpose_ct_ct64
conv2d_transpose_with_chan_pt_ct64 = shell_ops.conv2d_transpose_with_chan_pt_ct64
conv2d_transpose_with_chan_ct_pt64 = shell_ops.conv2d_transpose_with_chan_ct_pt64
conv2d_transpose_with_chan_ct_ct64 = shell_ops.conv2d_transpose_with_chan_ct_ct64
max_unpool2d_ct64 = shell_ops.max_unpool2d_ct64

# MPC-based ops.
clip_and_noise_features_party = shell_ops.clip_and_noise_features_party
clip_and_noise_labels_party = shell_ops.clip_and_noise_labels_party

# Distribution sampling ops.
sample_centered_gaussian_f64 = shell_ops.sample_centered_gaussian_f64
sample_centered_gaussian_l64 = shell_ops.sample_centered_gaussian_l64
