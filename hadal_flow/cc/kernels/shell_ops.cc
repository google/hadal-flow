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

#include "shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::OkStatus;
using tensorflow::TensorProto;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;
using tensorflow::shape_inference::ConcatShape;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ScalarShape;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::UnchangedShape;

// Tensorflow does not have size_t but Shell Context parameters require it.
// Code below must assume size_t is a unit64 because of this.
static_assert(sizeof(size_t) == sizeof(uint64_t), "Fatal size mismatch");

REGISTER_OP("ContextImport64")
    .Input("log_n: uint64")
    .Input("main_moduli: uint64")
    .Input("aux_moduli: uint64")
    .Input("plaintext_modulus: uint64")
    .Input("noise_variance: uint64")
    .Input("seed: string")
    .Output("shell_context: variant")
    .Output("new_log_n: uint64")
    .Output("new_qs: uint64")
    .Output("new_ps: uint64")
    .Output("new_pt_modulus: uint64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      c->set_output(2, c->input(1));
      c->set_output(3, c->input(2));
      c->set_output(4, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("AutoShellContext64")
    .Input("log2_cleartext_sz: uint64")
    .Input("scaling_factor: float")
    .Input("log2_noise_offset: int64")
    .Input("noise_variance: uint64")
    .Output("shell_context: variant")
    .Output("new_log_n: uint64")
    .Output("new_qs: uint64")
    .Output("new_ps: uint64")
    .Output("new_pt_modulus: uint64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      c->set_output(2, c->UnknownShape());
      c->set_output(3, c->UnknownShape());
      c->set_output(4, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("PolynomialImport64")
    .Attr(
        "Dtype: {uint8, int8, int16, uint16, int32, uint32, int64, uint64, "
        "float, double}")
    .Attr("scaling_factor: float")
    .Attr("random_round: bool")
    .Input("shell_context: variant")
    .Input("val: Dtype")
    .Output("out: variant")
    .SetShapeFn(ImportAndRemoveBatchingDimShape);

REGISTER_OP("PolynomialExport64")
    .Attr(
        "dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64, "
        "float, double}")
    .Attr("batching_dim: int")
    .Attr("scaling_factor: float")
    .Input("shell_context: variant")
    .Input("val: variant")
    .Input("runtime_batching_dim: int64")
    .Output("out: dtype")
    .SetShapeFn(ExportAndAddBatchingDimShape<1>);

REGISTER_OP("KeyGen64")
    .Input("context: variant")
    .Output("key: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("Encrypt64")
    .Input("context: variant")
    .Input("key: variant")
    .Input("val: variant")
    .Output("out: variant")
    .SetShapeFn(UnchangedArgShape<2>);

REGISTER_OP("Decrypt64")
    .Attr(
        "dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64, "
        "float, double}")
    .Attr("batching_dim: int")
    .Attr("scaling_factor: float")
    .Input("context: variant")
    .Input("key: variant")
    .Input("val: variant")
    .Input("runtime_batching_dim: int64")
    .Output("out: dtype")
    .SetShapeFn(ExportAndAddBatchingDimShape<2>);

// Add and subtract.
REGISTER_OP("AddCtCt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("AddCtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("AddPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("SubCtCt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("SubCtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("SubPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("NegCt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("negated_value: variant")
    .SetShapeFn(UnchangedArgShape<1>);

REGISTER_OP("NegPt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("negated_value: variant")
    .SetShapeFn(UnchangedArgShape<1>);

// Multiply.
REGISTER_OP("MulCtCt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("MulCtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("MulCtTfScalar64")
    .Attr(
        "Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64, "
        "float, double}")
    .Attr("scaling_factor: float")
    .Attr("random_round: bool")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: Dtype")
    .Output("c: variant")
    .SetShapeFn(UnchangedArgShape<1>);

REGISTER_OP("MulPtTfScalar64")
    .Attr(
        "Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64, "
        "float, double}")
    .Attr("scaling_factor: float")
    .Attr("random_round: bool")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: Dtype")
    .Output("c: variant")
    .SetShapeFn(UnchangedArgShape<1>);

REGISTER_OP("MulPtPt64")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: variant")
    .Output("c: variant")
    .SetShapeFn(ShellBroadcastingOpShape);

REGISTER_OP("MatMulCtPt64")
    .Attr("Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Input("context: variant")
    .Input("a: variant")
    .Input("b: Dtype")
    .Attr("reduce_dim_size: int")
    .Output("c: variant")
    .SetShapeFn(ShellMatMulCtPtShape);

REGISTER_OP("MatMulPtCt64")
    .Attr("Dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Input("context: variant")
    .Input("a: Dtype")
    .Input("b: variant")
    .Input("rotation_key: variant")
    .Attr("reduction: string")
    .Output("c: variant")
    .SetShapeFn(ShellMatMulPtCtShape);

// Rotate.
REGISTER_OP("RotationKeyGen64")
    .Input("context: variant")
    .Input("key: variant")
    .Output("rotation_key: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("Roll64")
    .Input("context: variant")
    .Input("rotation_key: variant")
    .Input("value: variant")
    .Input("shift: int64")
    .Output("rotated_value: variant")
    .SetShapeFn(UnchangedArgShape<2>);

REGISTER_OP("ReduceSumByRotationCt64")
    .Input("context: variant")
    .Input("rotation_key: variant")
    .Input("value: variant")
    .Output("repeated_reduce_sum: variant")
    .SetShapeFn(UnchangedArgShape<2>);

REGISTER_OP("ReduceSumWithModulusPt64")
    .Attr("dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64}")
    .Attr("axis: int")
    .Input("context: variant")
    .Input("value: dtype")
    .Output("reduced: dtype")
    .SetShapeFn([](InferenceContext* c) {
      tsl::int32 rank = c->Rank(c->input(1));
      if (rank == -1) {
        c->set_output(0, c->UnknownShape());
        return OkStatus();
      }

      tsl::int32 axis;
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));

      int clamped_axis = axis;
      if (clamped_axis < 0) {
        clamped_axis += rank;
      }
      if (clamped_axis < 0 || clamped_axis > rank) {
        return InvalidArgument("axis must be in the range [0, rank], got axis ",
                               clamped_axis, " and rank ", rank);
      }

      ShapeHandle output;

      // If this op ever supports keepdim=True, use the following shape.
      //  DimensionHandle reduced_dim = c->MakeDim({1});
      //  TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), clamped_axis,
      //  reduced_dim, &output));

      // This op currently only supports keepdim=False whose shape is computed
      // via the following.
      ShapeHandle prefix;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 0, clamped_axis, &prefix));

      ShapeHandle postfix;
      TF_RETURN_IF_ERROR(
          c->Subshape(c->input(1), clamped_axis + 1, rank, &postfix));

      if (clamped_axis == 0) {
        output = postfix;
      } else if (clamped_axis == rank - 1) {
        output = prefix;
      } else {
        TF_RETURN_IF_ERROR(c->Concatenate(prefix, postfix, &output));
      }

      c->set_output(0, output);
      return OkStatus();
    });

REGISTER_OP("FastRotationKeyGen64")
    .Input("context: variant")
    .Input("key: variant")
    .Output("fast_rotation_key: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("FastReduceSumByRotation64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("repeated_reduce_sum: variant")
    .SetShapeFn(UnchangedArgShape<1>);

REGISTER_OP("DecryptFastRotated64")
    .Attr(
        "dtype: {uint8, int8, uint16, int16, uint32, int32, uint64, int64, "
        "float, double}")
    .Attr("batching_dim: int")
    .Attr("scaling_factor: float")
    .Input("context: variant")
    .Input("fast_rotation_key: variant")
    .Input("val: variant")
    .Input("runtime_batching_dim: int64")
    .Output("out: dtype")
    .SetShapeFn(ExportAndAddBatchingDimShape<2>);

REGISTER_OP("ReduceSumCt64")
    .Input("context: variant")
    .Input("value: variant")
    .Attr("axis: int")
    .Attr("reduce_dim_size: int")
    .Output("repeated_reduce_sum: variant")
    .SetShapeFn([](InferenceContext* c) {
      tsl::int32 rank = c->Rank(c->input(1));

      tsl::int32 axis;
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));

      // Check that axis is in the correct range.
      if (axis == 0) {
        return InvalidArgument(
            "axis may not be zero. See ReduceSumByRotation()");
      }

      // Recall first dimension of a shell variant tensor is the packing
      // dimension.
      int clamped_axis = axis;
      if (clamped_axis < 0) {
        clamped_axis += rank + 1;
      } else if (clamped_axis > 0) {
        clamped_axis -= 1;
      }

      if (clamped_axis < 0 || clamped_axis > rank) {
        return InvalidArgument("axis must be in the range [0, rank], got ",
                               clamped_axis);
      }

      ShapeHandle output;

      // If this op ever supports keepdim=True, use the following shape.
      //  DimensionHandle reduced_dim = c->MakeDim({1});
      //  TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), clamped_axis,
      //  reduced_dim, &output));

      // This op currently only supports keepdim=False whose shape is computed
      // via the following.
      ShapeHandle prefix;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 0, clamped_axis, &prefix));

      ShapeHandle postfix;
      TF_RETURN_IF_ERROR(
          c->Subshape(c->input(1), clamped_axis + 1, rank, &postfix));

      if (clamped_axis == 0) {
        output = postfix;
      } else if (clamped_axis == rank - 1) {
        output = prefix;
      } else {
        TF_RETURN_IF_ERROR(c->Concatenate(prefix, postfix, &output));
      }

      c->set_output(0, output);
      return OkStatus();
    });

// Modulus switching.
REGISTER_OP("ModulusReduceContext64")
    .Input("context: variant")
    .Output("reduced_context: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("ModulusReduceKey64")
    .Input("unreduced_context: variant")
    .Input("key: variant")
    .Output("reduced_key: variant")
    .SetShapeFn(ScalarShape);

REGISTER_OP("ModulusReduceCt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("reduced_value: variant")
    .SetShapeFn(UnchangedArgShape<1>);

REGISTER_OP("ModulusReducePt64")
    .Input("context: variant")
    .Input("value: variant")
    .Output("reduced_value: variant")
    .SetShapeFn(UnchangedArgShape<1>);

// Shape kernels.
REGISTER_OP("ExpandDimsVariant")
    .Input("value: variant")
    .Attr("axis: int")
    .Output("expanded_value: variant")
    .SetShapeFn([](InferenceContext* c) {
      tsl::int32 rank = c->Rank(c->input(0));
      if (rank == InferenceContext::kUnknownRank) {
        c->set_output(0, c->UnknownShape());
        return OkStatus();
      }

      tsl::int32 axis;
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));

      tsl::int32 clamped_axis = axis;
      if (clamped_axis < 0) {
        clamped_axis += rank + 1;  // + 1 for packing dimension.
      } else if (clamped_axis > 0) {
        clamped_axis -= 1;  // -1 for packing dimension.
      }

      // Check that axis is in the correct range.
      if (clamped_axis < 0 || clamped_axis > rank) {
        return InvalidArgument("expand_dims axis must be in the range [-", rank,
                               ", ", rank, "]. Got ", axis);
      }

      ShapeHandle prefix;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 0, clamped_axis, &prefix));

      ShapeHandle postfix;
      TF_RETURN_IF_ERROR(
          c->Subshape(c->input(0), clamped_axis, rank, &postfix));

      ShapeHandle output;
      ShapeHandle axis_dim = c->MakeShape({1});
      TF_RETURN_IF_ERROR(c->Concatenate(prefix, axis_dim, &output));
      TF_RETURN_IF_ERROR(c->Concatenate(output, postfix, &output));

      c->set_output(0, output);
      return OkStatus();
    });

REGISTER_OP("ConcatCt64")
    .Input("axis: int32")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      return ConcatShape(c, c->num_inputs() - 1);
    });

REGISTER_OP("ConcatPt64")
    .Input("axis: int32")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      return ConcatShape(c, c->num_inputs() - 1);
    });

// Segment sum where the segment_ids are plaintexts.
// Based on :
// https://github.com/tensorflow/tensorflow/blob/dfdba938a0048611319ce192d8f17639e058ad00/tensorflow/core/ops/math_ops.cc#L1293
REGISTER_OP("UnsortedCtSegmentSum")
    .Input("shell_context: variant")
    .Input("data: variant")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Input("rotation_key: variant")
    .Attr("reduction: string")
    .Output("output: variant")
    .Output("reduction_counts: Tindices")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(ShellSegmentReductionWithNumSegmentsShape);

// Convolutions.
REGISTER_OP("Conv2dPtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2d);

REGISTER_OP("Conv2dCtPt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2d);

REGISTER_OP("Conv2dCtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2d);

REGISTER_OP("Conv2dWithChanPtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dWithChan);

REGISTER_OP("Conv2dWithChanCtPt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dWithChan);

REGISTER_OP("Conv2dWithChanCtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dWithChan);

REGISTER_OP("Conv2dTransposePtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dTranspose);

REGISTER_OP("Conv2dTransposeCtPt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dTranspose);

REGISTER_OP("Conv2dTransposeCtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dTranspose);

REGISTER_OP("Conv2dTransposeWithChanPtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dTransposeWithChan);

REGISTER_OP("Conv2dTransposeWithChanCtPt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dTransposeWithChan);

REGISTER_OP("Conv2dTransposeWithChanCtCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("filter: variant")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("dilations: list(int)")
    .Attr("filter_num_elements: int")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn(ShellConv2dTransposeWithChan);

REGISTER_OP("MaxUnpool2dCt64")
    .Input("shell_context: variant")
    .Input("x: variant")
    .Input("argmax: int64")
    .Attr("pool_size: list(int)")
    .Attr("strides: list(int)")
    .Attr("padding: list(int)")
    .Attr("output_shape: list(int)")
    .Output("output: variant")
    .SetShapeFn([](InferenceContext* c) {
      return ShapeFromAttr(c, "output_shape", 0, true);
    });

// MPC-based kernels.
REGISTER_OP("ClipAndNoiseFeaturesParty")
    .Attr("Dtype: {int32, int64}")
    .Attr("Bitwidth: int")
    .Attr("StartPort: int")
    .Attr("LabelPartyHost: string")
    .Input("mask: Dtype")
    .Output("clipped_noised_grad: Dtype")
    .SetShapeFn(UnchangedArgShape<0>)
    .SetIsStateful();  // For port allocations.

REGISTER_OP("ClipAndNoiseLabelsParty")
    .Attr("Dtype: {int32, int64}")
    .Attr("Bitwidth: int")
    .Attr("StartPort: int")
    .Attr("FeaturePartyHost: string")
    .Input("masked_grads: Dtype")
    .Input("clipping_thresh: Dtype")
    .Input("noise: Dtype")
    .SetShapeFn(ScalarShape)
    .SetIsStateful();  // For port allocations.

// Distribution sampling kernels.
REGISTER_OP("SampleCenteredGaussianF64")
    .Input("scale: float")
    .Attr("base_scale: float")
    .Attr("max_scale: float")
    .Output("a: int64")
    .Output("b: int64")
    .SetIsStateful()  // Prevent caching output.
    .SetShapeFn(ShellSampleCenteredGaussianF);

REGISTER_OP("SampleCenteredGaussianL64")
    .Input("shell_context: variant")
    .Input("num_samples: int64")
    .Attr("base_scale: float")
    .Attr("max_scale: float")
    .Output("samples: int64")
    .SetIsStateful()  // Prevent caching output.
    .SetShapeFn(ShellSampleCenteredGaussianL);

REGISTER_OP("SaveShellTensor")
    .Input("data: T")
    .Input("path: string")
    .Output("out_path: string")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return OkStatus();
    });

REGISTER_OP("LoadShellTensor")
    .Input("path: string")
    .Output("data: T")
    .Attr("T: type")
    .Attr("output_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      tensorflow::TensorShapeProto shape_proto;
      if (c->GetAttr("output_shape", &shape_proto).ok() &&
          !shape_proto.unknown_rank()) {
        ShapeHandle shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &shape));
        c->set_output(0, shape);
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return OkStatus();
    });
