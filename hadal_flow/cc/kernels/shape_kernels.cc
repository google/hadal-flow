/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "polynomial_variant.h"
#include "symmetric_variants.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using tensorflow::DEVICE_CPU;
using tensorflow::DT_VARIANT;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::OpInputList;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::Internal;
using tensorflow::errors::InvalidArgument;

// This class is exactly like TensorFlow's ExpandDimsOp, but allows operating
// on a tensor with a variant dtype.
class ExpandDimsVariantOp : public OpKernel {
 private:
  int dim;

 public:
  explicit ExpandDimsVariantOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {
    // Get the dimension to expand from the op attributes.
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("axis", &dim));

    // Recall first dimension of a shell variant tensor is the packing
    // dimension. We don't allow expanding this dimension.
    OP_REQUIRES(op_ctx, dim != 0, InvalidArgument("Invalid dimension index."));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, dim != 0, InvalidArgument("Invalid dimension index."));

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    int clamped_dim = dim;
    if (clamped_dim < 0) {
      clamped_dim += ctx->input(0).dims() + 1;  // + 1 for packing dim.
    } else if (clamped_dim > 0) {
      clamped_dim -= 1;  // -1 for packing dimension.
    }

    OP_REQUIRES(ctx, clamped_dim >= 0 && clamped_dim <= ctx->input(0).dims(),
                InvalidArgument("Tried to expand dim index ", clamped_dim,
                                " for tensor with ", ctx->input(0).dims(),
                                " dimensions."));

    auto existing_dims = ctx->input(0).shape().dim_sizes();
    // Safe - # elements in tensor dims bounded.
    int const existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64> new_shape(existing_dims_size);
    for (size_t i = 0; i < new_shape.size(); ++i) {
      new_shape[i] = existing_dims[i];
    }

    // Clamp to the end if needed.
    clamped_dim = std::min<int32>(clamped_dim, existing_dims_size);
    new_shape.emplace(new_shape.begin() + clamped_dim, 1);
    TensorShape const output_shape(new_shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &output));
    if (!output->CopyFrom(ctx->input(0), output_shape)) {
      // This should never happen, since the sizes of the input and output
      // should always be the same (we only expand the dimension with 1).
      ctx->SetStatus(Internal("Could not expand dimension with input shape ",
                              ctx->input(0).shape().DebugString(),
                              " and output shape ",
                              output_shape.DebugString()));
    }
  }

  bool IsExpensive() override { return false; }
};

template <typename CtOrPolyVariant>
class ConcatVariantOp : public OpKernel {
 public:
  explicit ConcatVariantOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Get the axis to concatenate along.
    OP_REQUIRES_VALUE(int32 axis, ctx, GetScalar<int32>(ctx, 0));

    // Get the list of tensors to concatenate.
    OpInputList values;
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values));

    // Get the number of tensors to concatenate.
    int const N = values.size();

    // Check that all input tensors have the same shape, except for the
    // dimension to concatenate along.
    TensorShape output_shape = values[0].shape();
    int64 concat_dim_size = 0;
    for (int i = 0; i < N; ++i) {
      OP_REQUIRES(ctx, values[i].dtype() == DT_VARIANT,
                  InvalidArgument("All inputs must be variant tensors."));
      for (int d = 0; d < values[i].dims(); ++d) {
        if (d == axis) {
          concat_dim_size += values[i].dim_size(d);
        } else {
          OP_REQUIRES(ctx, values[i].dim_size(d) == output_shape.dim_size(d),
                      InvalidArgument(
                          "All input tensors must have the same shape, except "
                          "for the dimension to concatenate along."));
        }
      }
    }
    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (axis < 0) {
      axis += output_shape.dims();
    }
    OP_REQUIRES(ctx, axis >= 0 && axis < output_shape.dims(),
                InvalidArgument("Invalid axis to concat over. Must be in the "
                                "range [0, ",
                                output_shape.dims(), ")."));

    output_shape.set_dim(axis, concat_dim_size);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat_inner_outer_dims<Variant>(axis - 1);

    // Concatenate the input tensors along the specified axis.
    int64 offset = 0;
    for (int i = 0; i < N; ++i) {
      // Get the input tensor as a chip using flat_inner_outer_dims() with the
      // axis to concatenate along as the middle dimension.
      auto input_flat = values[i].flat_inner_outer_dims<Variant>(axis - 1);

      // Copy the input tensor to the output tensor.
      for (int64 j = 0; j < input_flat.dimension(0); ++j) {
        for (int64 k = 0; k < input_flat.dimension(1); ++k) {
          for (int64 l = 0; l < input_flat.dimension(2); ++l) {
            // Lock the mutex while copying.
            CtOrPolyVariant const* ct_var =
                input_flat(j, k, l).get<CtOrPolyVariant>();
            // TODO if debug
            OP_REQUIRES(ctx, ct_var != nullptr,
                        InvalidArgument(
                            "ConcatVariantOp: SymmetricCtVariant or "
                            "PolynomialVariant did not unwrap successfully."));
            std::lock_guard<std::mutex> lock(
                const_cast<CtOrPolyVariant*>(ct_var)->mutex.mutex);
            flat_output(j, offset + k, l) = input_flat(j, k, l);
          }
        }
      }

      // Update the offset to start at the next chip for subsequent input
      // tensor.
      offset += input_flat.dimension(1);
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("ExpandDimsVariant").Device(DEVICE_CPU),
                        ExpandDimsVariantOp);

REGISTER_KERNEL_BUILDER(Name("ConcatCt64").Device(DEVICE_CPU),
                        ConcatVariantOp<SymmetricCtVariant<uint64_t>>);
REGISTER_KERNEL_BUILDER(Name("ConcatPt64").Device(DEVICE_CPU),
                        ConcatVariantOp<PolynomialVariant<uint64_t>>);
