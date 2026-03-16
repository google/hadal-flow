// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/common_shape_fns.h"
// #include "tensorflow/core/framework/op.h"
#include "discrete_gaussian_sampler.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::OkStatus;
using tensorflow::Status;
using tensorflow::errors::InvalidArgument;
using tensorflow::shape_inference::BroadcastBinaryOpOutputShapeFnHelper;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ScalarShape;
using tensorflow::shape_inference::ShapeHandle;

Status ImportAndRemoveBatchingDimShape(InferenceContext* c) {
  ShapeHandle output;

  // First dimension of "in" is stored via shell Polynomial.
  TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 1, &output));

  c->set_output(0, output);
  return OkStatus();
}

Status ShellBroadcastingOpShape(InferenceContext* c) {
  if (c->num_inputs() != 3) {
    return InvalidArgument("Expected 3 inputs but got: ", c->num_inputs());
  }

  ShapeHandle a_shape = c->input(1);
  ShapeHandle b_shape = c->input(2);
  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(c, a_shape, b_shape,
                                                          true, &output_shape));
  c->set_output(0, output_shape);
  return OkStatus();
}

Status ShellMatMulCtPtShape(InferenceContext* c) {
  ShapeHandle a_shape;  // a is the ciphertext with batch axis packing.
  ShapeHandle b_shape;  // b is the plaintext.
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &a_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &b_shape));

  // When the ciphertext a is rank 1, it is still considered a matrix
  // because the first axis is the packing dimension.
  bool a_batched = c->Rank(a_shape) > 1;

  // Determine output rows and columns.
  DimensionHandle output_rows =
      a_batched ? c->Dim(a_shape, -2) : c->MakeDim({1});
  DimensionHandle output_cols = c->Dim(b_shape, -1);

  // Inner dimensions should be compatible.
  DimensionHandle inner_merged;
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(a_shape, -1), c->Dim(b_shape, -2), &inner_merged));

  // Batch dimensions should broadcast with each other.
  ShapeHandle a_batch_shape;
  ShapeHandle b_batch_shape;
  ShapeHandle output_batch_shape;
  TF_RETURN_IF_ERROR(
      c->Subshape(a_shape, 0, a_batched ? -2 : -1, &a_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_shape));

  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, a_batch_shape, b_batch_shape, true, &output_batch_shape));

  ShapeHandle output_shape;
  if (a_batched) {
    TF_RETURN_IF_ERROR(c->Concatenate(output_batch_shape,
                                      c->Matrix(output_rows, output_cols),
                                      &output_shape));
  } else {
    // If a is not batched, the first matrix dimension is the packing dimension.
    TF_RETURN_IF_ERROR(c->Concatenate(output_batch_shape,
                                      c->Vector(output_cols), &output_shape));
  }

  c->set_output(0, output_shape);
  return OkStatus();
}

Status ShellMatMulPtCtShape(InferenceContext* c) {
  ShapeHandle a_shape;  // a is the plaintext.
  ShapeHandle b_shape;  // b is the ciphertext with batch axis packing.
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &a_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &b_shape));

  // Determine output rows and columns.
  DimensionHandle output_rows = c->Dim(a_shape, -2);
  DimensionHandle output_cols = c->Dim(b_shape, -1);

  // Inner dimensions compatibility checked at runtime.
  ShapeHandle output_batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &output_batch_shape));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(
      output_batch_shape, c->Matrix(output_rows, output_cols), &output_shape));

  c->set_output(0, output_shape);
  return OkStatus();
}

// Based on
// https://github.com/tensorflow/tensorflow/blob/c6e9fd55508e466aa7db265eb5742ac9e4c4332e/tensorflow/core/framework/common_shape_fns.cc#L2408
Status ShellSegmentReductionWithNumSegmentsShape(InferenceContext* c) {
  ShapeHandle s_data = c->input(1);
  ShapeHandle s_segment_ids = c->input(2);
  ShapeHandle s_num_segments = c->input(3);
  TF_RETURN_IF_ERROR(c->WithRank(s_num_segments, 0, &s_num_segments));

  ShapeHandle data_out;
  ShapeHandle reduction_counters;

  if (c->RankKnown(s_segment_ids)) {
    // Leading dimensions of data must be compatible with dimensions of
    // s_segment_ids, but ignore the batch axis packing dimension which
    // is checked at op kernel time.
    ShapeHandle s_segment_ids_suffix;
    TF_RETURN_IF_ERROR(c->Subshape(s_segment_ids, 1, &s_segment_ids_suffix));
    ShapeHandle matching_prefix;
    TF_RETURN_IF_ERROR(c->MergePrefix(s_data, s_segment_ids_suffix, &s_data,
                                      &s_segment_ids_suffix));

    // Get the value of the num_segments input tensor.
    DimensionHandle num_segments_dim;
    TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(3, &num_segments_dim));

    std::string reduction_type;
    TF_RETURN_IF_ERROR(c->GetAttr("reduction", &reduction_type));

    // When the reduction type is galois, the output shape is:
    // [2, segment_id_rank, s_data[segment_id_rank - 1:]].
    // 2 is because the top and bottom of the ciphertexts are treated
    // independently. The packing dimension is not included as the output is
    // a ciphertext and holds this dimension implicitly.
    // When the reduction type is none, the 2 is omitted.
    ShapeHandle s_data_suffix;
    auto rank = c->Rank(s_segment_ids_suffix);
    TF_RETURN_IF_ERROR(c->Subshape(s_data, rank, &s_data_suffix));

    TF_RETURN_IF_ERROR(
        c->Concatenate(c->Vector(num_segments_dim), s_data_suffix, &data_out));
    if (reduction_type == "galois") {
      TF_RETURN_IF_ERROR(
          c->Concatenate(c->Vector(c->MakeDim(2)), data_out, &data_out));
    }

    TF_RETURN_IF_ERROR(c->WithRankAtLeast(s_segment_ids, 1, &s_segment_ids));
    DimensionHandle num_slots = c->Dim(s_segment_ids, 0);
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(num_slots),
                                      c->Vector(num_segments_dim),
                                      &reduction_counters));

  } else {
    data_out = c->UnknownShape();
    reduction_counters = c->UnknownShape();
  }

  c->set_output(0, data_out);
  c->set_output(1, reduction_counters);
  return OkStatus();
}

Status ShellConv2dImpl(InferenceContext* c, bool different_num_in_channels) {
  // If the output shape is provided, return that.
  std::vector<tsl::int32> requested_output_shape;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shape", &requested_output_shape));
  if (!requested_output_shape.empty()) {
    ShapeHandle output_shape_handle = c->Scalar();
    // Skip the batching dimension.
    for (size_t i = 1; i < requested_output_shape.size(); i++) {
      DimensionHandle h = c->MakeDim(requested_output_shape[i]);
      TF_RETURN_IF_ERROR(c->Concatenate(output_shape_handle, c->Vector(h),
                                        &output_shape_handle));
    }
    c->set_output(0, output_shape_handle);
    return OkStatus();
  }

  // Input shape s_x is {height, width, in_channels}. Output shape is
  // {out_height, out_width, out_channels}. The batch size is implicit in the
  // ciphertext ring degree and not part of the shape.
  ShapeHandle s_x = c->input(1);
  ShapeHandle s_filter = c->input(2);
  DimensionHandle one = c->MakeDim(1);

  // Check that the input tensor has rank 3.
  TF_RETURN_IF_ERROR(c->WithRank(s_x, 3, &s_x));
  DimensionHandle const height = c->Dim(s_x, 0);
  DimensionHandle const width = c->Dim(s_x, 1);
  DimensionHandle const in_channels = c->Dim(s_x, 2);

  // Check that the filter tensor has rank 4.
  TF_RETURN_IF_ERROR(c->WithRank(s_filter, 4, &s_filter));
  DimensionHandle const filter_height = c->Dim(s_filter, 0);
  DimensionHandle const filter_width = c->Dim(s_filter, 1);
  DimensionHandle const filter_in_channels = c->Dim(s_filter, 2);
  DimensionHandle const filter_out_channels = c->Dim(s_filter, 3);

  // Check the stride.
  std::vector<tsl::int32> stride;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &stride));
  // DimensionHandle const stride_batch = c->MakeDim(stride[0]);
  DimensionHandle const stride_height = c->MakeDim(stride[1]);
  DimensionHandle const stride_width = c->MakeDim(stride[2]);
  DimensionHandle const stride_in_channels = c->MakeDim(stride[3]);

  // Check the padding tensor.
  std::vector<tsl::int32> padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
  DimensionHandle const padding_top = c->MakeDim(padding[0]);
  DimensionHandle const padding_bottom = c->MakeDim(padding[1]);
  DimensionHandle const padding_left = c->MakeDim(padding[2]);
  DimensionHandle const padding_right = c->MakeDim(padding[3]);

  std::vector<tsl::int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));
  // DimensionHandle const dilation_batch = c->MakeDim(dilations[0]);
  DimensionHandle const dilation_height = c->MakeDim(dilations[1]);
  DimensionHandle const dilation_width = c->MakeDim(dilations[2]);
  // DimensionHandle const dilation_channel = c->MakeDim(dilations[3]);

  // Prepare dilated filter dimensions.
  DimensionHandle filter_dilated_height = filter_height;
  TF_RETURN_IF_ERROR(
      c->Subtract(filter_dilated_height, one, &filter_dilated_height));
  TF_RETURN_IF_ERROR(c->Multiply(filter_dilated_height, dilation_height,
                                 &filter_dilated_height));
  TF_RETURN_IF_ERROR(
      c->Add(filter_dilated_height, one, &filter_dilated_height));

  DimensionHandle filter_dilated_width = filter_width;
  TF_RETURN_IF_ERROR(
      c->Subtract(filter_dilated_width, one, &filter_dilated_width));
  TF_RETURN_IF_ERROR(
      c->Multiply(filter_dilated_width, dilation_width, &filter_dilated_width));
  TF_RETURN_IF_ERROR(c->Add(filter_dilated_width, one, &filter_dilated_width));

  // Add the padding to the height and width.
  DimensionHandle out_height = height;
  TF_RETURN_IF_ERROR(c->Add(out_height, padding_bottom, &out_height));
  TF_RETURN_IF_ERROR(c->Add(out_height, padding_top, &out_height));
  TF_RETURN_IF_ERROR(
      c->Subtract(out_height, filter_dilated_height, &out_height));
  TF_RETURN_IF_ERROR(c->Divide(out_height, stride_height, false, &out_height));
  TF_RETURN_IF_ERROR(c->Add(out_height, one, &out_height));

  DimensionHandle out_width = width;
  TF_RETURN_IF_ERROR(c->Add(out_width, padding_right, &out_width));
  TF_RETURN_IF_ERROR(c->Add(out_width, padding_left, &out_width));
  TF_RETURN_IF_ERROR(c->Subtract(out_width, filter_dilated_width, &out_width));
  TF_RETURN_IF_ERROR(c->Divide(out_width, stride_width, false, &out_width));
  TF_RETURN_IF_ERROR(c->Add(out_width, one, &out_width));

  DimensionHandle channels = in_channels;
  TF_RETURN_IF_ERROR(c->Subtract(channels, filter_in_channels, &channels));
  TF_RETURN_IF_ERROR(c->Divide(channels, stride_in_channels, true, &channels));
  TF_RETURN_IF_ERROR(c->Add(channels, one, &channels));

  ShapeHandle output_shape = c->Scalar();
  TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(out_height), c->Vector(out_width),
                                    &output_shape));
  if (different_num_in_channels) {
    // If the number of input channels in the input image and the filter can be
    // different, the output shape must include the channel dimension, i.e.,
    // {out_height, out_width, channels, out_channels}.
    // Otherwise, channels is excluded, i.e.
    // {out_height, out_width, out_channels}.
    TF_RETURN_IF_ERROR(
        c->Concatenate(output_shape, c->Vector(channels), &output_shape));
  }
  TF_RETURN_IF_ERROR(c->Concatenate(
      output_shape, c->Vector(filter_out_channels), &output_shape));

  c->set_output(0, output_shape);
  return OkStatus();
}

Status ShellConv2d(InferenceContext* c) { return ShellConv2dImpl(c, false); }

Status ShellConv2dWithChan(InferenceContext* c) {
  return ShellConv2dImpl(c, true);
}

Status ShellConv2dTransposeImpl(InferenceContext* c,
                                bool different_num_in_channels) {
  // If the output shape is provided, return that.
  std::vector<tsl::int32> requested_output_shape;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shape", &requested_output_shape));
  if (!requested_output_shape.empty()) {
    ShapeHandle output_shape_handle = c->Scalar();
    // Skip the batching dimension.
    for (size_t i = 1; i < requested_output_shape.size(); i++) {
      DimensionHandle h = c->MakeDim(requested_output_shape[i]);
      TF_RETURN_IF_ERROR(c->Concatenate(output_shape_handle, c->Vector(h),
                                        &output_shape_handle));
    }
    c->set_output(0, output_shape_handle);
    return OkStatus();
  }

  // Input shape s_x is {height, width, in_channels}. Output shape is
  // {out_height, out_width, out_channels}. The batch size is implicit in the
  // ciphertext ring degree and not part of the shape.
  ShapeHandle s_x = c->input(1);
  ShapeHandle s_filter = c->input(2);
  DimensionHandle one = c->MakeDim(1);

  // Check that the input tensor has rank 3.
  TF_RETURN_IF_ERROR(c->WithRank(s_x, 3, &s_x));
  DimensionHandle const height = c->Dim(s_x, 0);
  DimensionHandle const width = c->Dim(s_x, 1);
  DimensionHandle const in_channels = c->Dim(s_x, 2);

  // Check that the filter tensor has rank 4.
  TF_RETURN_IF_ERROR(c->WithRank(s_filter, 4, &s_filter));
  DimensionHandle const filter_height = c->Dim(s_filter, 0);
  DimensionHandle const filter_width = c->Dim(s_filter, 1);
  DimensionHandle const filter_out_channels = c->Dim(s_filter, 2);
  DimensionHandle const filter_in_channels = c->Dim(s_filter, 3);

  // Check the stride.
  std::vector<tsl::int32> stride;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &stride));
  // DimensionHandle const stride_batch = c->MakeDim(stride[0]);
  DimensionHandle const stride_height = c->MakeDim(stride[1]);
  DimensionHandle const stride_width = c->MakeDim(stride[2]);
  DimensionHandle const stride_in_channels = c->MakeDim(stride[3]);

  // Check the padding tensor.
  std::vector<tsl::int32> padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
  DimensionHandle const padding_top = c->MakeDim(padding[0]);
  DimensionHandle const padding_bottom = c->MakeDim(padding[1]);
  DimensionHandle const padding_left = c->MakeDim(padding[2]);
  DimensionHandle const padding_right = c->MakeDim(padding[3]);

  // Add the padding to the height and width.
  DimensionHandle out_height = height;
  TF_RETURN_IF_ERROR(c->Subtract(out_height, one, &out_height));
  TF_RETURN_IF_ERROR(c->Multiply(out_height, stride_height, &out_height));
  TF_RETURN_IF_ERROR(c->Add(out_height, filter_height, &out_height));
  TF_RETURN_IF_ERROR(c->Subtract(out_height, padding_bottom, &out_height));
  TF_RETURN_IF_ERROR(c->Subtract(out_height, padding_top, &out_height));

  DimensionHandle out_width = width;
  TF_RETURN_IF_ERROR(c->Subtract(out_width, one, &out_width));
  TF_RETURN_IF_ERROR(c->Multiply(out_width, stride_width, &out_width));
  TF_RETURN_IF_ERROR(c->Add(out_width, filter_width, &out_width));
  TF_RETURN_IF_ERROR(c->Subtract(out_width, padding_right, &out_width));
  TF_RETURN_IF_ERROR(c->Subtract(out_width, padding_left, &out_width));

  DimensionHandle channels = in_channels;
  TF_RETURN_IF_ERROR(c->Subtract(channels, one, &channels));
  TF_RETURN_IF_ERROR(c->Multiply(channels, stride_in_channels, &channels));
  TF_RETURN_IF_ERROR(c->Add(channels, filter_in_channels, &channels));

  ShapeHandle output_shape = c->Scalar();
  TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(out_height), c->Vector(out_width),
                                    &output_shape));
  if (different_num_in_channels) {
    // If the number of input channels in the input image and the filter can be
    // different, the output shape must include the channel dimension, i.e.,
    // {out_height, out_width, channels, out_channels}.
    // Otherwise, channels is excluded, i.e.
    // {out_height, out_width, out_channels}.
    TF_RETURN_IF_ERROR(
        c->Concatenate(output_shape, c->Vector(channels), &output_shape));
  }
  TF_RETURN_IF_ERROR(c->Concatenate(
      output_shape, c->Vector(filter_out_channels), &output_shape));

  c->set_output(0, output_shape);
  return OkStatus();
}

Status ShellConv2dTranspose(InferenceContext* c) {
  return ShellConv2dTransposeImpl(c, false);
}

Status ShellConv2dTransposeWithChan(InferenceContext* c) {
  return ShellConv2dTransposeImpl(c, true);
}

Status ShellSampleCenteredGaussianF(InferenceContext* c) {
  float base_scale;
  float max_scale;
  TF_RETURN_IF_ERROR(c->GetAttr("base_scale", &base_scale));
  TF_RETURN_IF_ERROR(c->GetAttr("max_scale", &max_scale));

  int n_max = 0, s_max = 0;
  TF_ASSIGN_OR_RETURN(
      std::tie(n_max, s_max),
      DiscreteGaussianSampler<uint64_t>::NumIterations(max_scale, base_scale));
  // The number of samples which must be scaled is one larger than the number
  // of iterations
  n_max += 1;

  ShapeHandle output_shape = c->Vector(c->MakeDim(n_max));
  c->set_output(0, output_shape);
  c->set_output(1, output_shape);
  return OkStatus();
}

Status ShellSampleCenteredGaussianL(InferenceContext* c) {
  float base_scale;
  float max_scale;

  TF_RETURN_IF_ERROR(c->GetAttr("base_scale", &base_scale));
  TF_RETURN_IF_ERROR(c->GetAttr("max_scale", &max_scale));

  int n_max = 0, s_max = 0;
  TF_ASSIGN_OR_RETURN(
      std::tie(n_max, s_max),
      DiscreteGaussianSampler<uint64_t>::NumIterations(max_scale, base_scale));
  // The number of samples which must be scaled is one larger than the number
  // of iterations
  n_max += 1;

  ShapeHandle output_shape = c->Matrix(c->UnknownDim(), c->MakeDim(n_max));
  c->set_output(0, output_shape);
  return OkStatus();
}

Status ShapeFromAttr(InferenceContext* c, char const* attr_name, int output_idx,
                     bool skip_batching_dim) {
  std::vector<tsl::int32> shape;
  TF_RETURN_IF_ERROR(c->GetAttr(attr_name, &shape));

  int shape_starts_at = skip_batching_dim ? 1 : 0;

  ShapeHandle output_shape = c->Vector(c->MakeDim(shape[shape_starts_at]));
  for (uint i = shape_starts_at + 1; i < shape.size(); i++) {
    auto const& dim = shape[i];
    TF_RETURN_IF_ERROR(c->Concatenate(output_shape, c->Vector(c->MakeDim(dim)),
                                      &output_shape));
  }

  c->set_output(output_idx, output_shape);
  return OkStatus();
}