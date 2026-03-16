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

#include "context_variant.h"
#include "shell_encryption/context.h"
#include "shell_encryption/montgomery.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "utils.h"

using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::tstring;
using tensorflow::uint64;
using tensorflow::Variant;

template <typename T>
class ContextImportOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;

 public:
  explicit ContextImportOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack inputs.
    OP_REQUIRES_VALUE(uint64_t log_n, op_ctx, GetScalar<uint64_t>(op_ctx, 0));
    OP_REQUIRES_VALUE(std::vector<T> qs, op_ctx, GetVector<T>(op_ctx, 1));
    OP_REQUIRES_VALUE(std::vector<T> ps, op_ctx, GetVector<T>(op_ctx, 2));
    OP_REQUIRES_VALUE(T pt_modulus, op_ctx, GetScalar<T>(op_ctx, 3));
    OP_REQUIRES_VALUE(size_t noise_variance, op_ctx,
                      GetScalar<size_t>(op_ctx, 4));
    OP_REQUIRES_VALUE(tstring t_seed, op_ctx, GetScalar<tstring>(op_ctx, 5));
    std::string seed(t_seed.data(), t_seed.size());

    // Allocate the outputs.
    Tensor* out0;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out0));
    Tensor* out1;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(1, TensorShape{}, &out1));
    Tensor* out2;
    OP_REQUIRES_OK(op_ctx,
                   op_ctx->allocate_output(
                       2, TensorShape{static_cast<int64_t>(qs.size())}, &out2));
    Tensor* out3;
    OP_REQUIRES_OK(op_ctx,
                   op_ctx->allocate_output(
                       3, TensorShape{static_cast<int64_t>(ps.size())}, &out3));
    Tensor* out4;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(4, TensorShape{}, &out4));

    // Initialize the context variant and store it in the output.
    ContextVariant<T> ctx_variant{};
    OP_REQUIRES_OK(op_ctx, ctx_variant.Initialize(log_n, qs, ps, pt_modulus,
                                                  noise_variance, seed));
    out0->scalar<Variant>()() = std::move(ctx_variant);

    // Output other parameters for usage with auto-context.
    out1->scalar<uint64_t>()() = log_n;
    for (size_t i = 0; i < qs.size(); ++i) {
      out2->flat<T>()(i) = qs[i];
    }
    for (size_t i = 0; i < ps.size(); ++i) {
      out3->flat<T>()(i) = ps[i];
    }
    out4->scalar<T>()() = pt_modulus;
  }
};

template <typename T>
class AutoContextOp : public OpKernel {
 public:
  explicit AutoContextOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // This is a placeholder for the AutoContextOp implementation used to
    // automatically choose encryption parameters for a given computation.
    // This op should be replaced with a ContextImportOp at graph optimization
    // time and should never be called.
    op_ctx->SetStatus(tensorflow::errors::Internal(
        "AutoContextOp should never be called. The "
        "graph should be optimized to replace this "
        "op with a ContextImportOp. If you see this, "
        "there was likely an error in generating "
        "ciphertext parameters for this computation."));
  }
};

REGISTER_KERNEL_BUILDER(Name("ContextImport64").Device(DEVICE_CPU),
                        ContextImportOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("AutoShellContext64").Device(DEVICE_CPU),
                        AutoContextOp<uint64>);

typedef ContextVariant<uint64> ContextVariantUint64;
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(ContextVariantUint64,
                                       ContextVariantUint64::kTypeName);