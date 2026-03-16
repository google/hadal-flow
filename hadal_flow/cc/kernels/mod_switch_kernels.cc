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
#include "polynomial_variant.h"
#include "shell_encryption/context.h"
#include "shell_encryption/montgomery.h"
#include "symmetric_variants.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
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
class ModulusReduceContextOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsRlweSecretKey<ModularInt>;
  using Gadget = rlwe::RnsGadget<ModularInt>;

 public:
  explicit ModulusReduceContextOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* context_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    auto reduced_qs = context_var->qs_;
    reduced_qs.pop_back();

    // Create a new context with the reduced moduli.
    ContextVariant<T> context_var_reduced{};
    OP_REQUIRES_OK(op_ctx,
                   context_var_reduced.Initialize(
                       context_var->log_n_, reduced_qs, context_var->ps_,
                       context_var->pt_modulus_, context_var->noise_variance_,
                       context_var->seed_));

    // Allocate a scalar output tensor and store the reduced context.
    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));
    out->scalar<Variant>()() = std::move(context_var_reduced);
  }
};

template <typename T>
class ModulusReduceKeyOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;

 public:
  explicit ModulusReduceKeyOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    // The first argument is the un-reduced context object. This is needed
    // to lazy decode the key before reducing it.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    OP_REQUIRES(op_ctx, shell_ctx_var != nullptr,
                InvalidArgument("ContextVariant did not unwrap successfully."));

    OP_REQUIRES_VALUE(SymmetricKeyVariant<T> const* secret_key_var, op_ctx,
                      GetVariant<SymmetricKeyVariant<T>>(op_ctx, 1));
    OP_REQUIRES(
        op_ctx, secret_key_var != nullptr,
        InvalidArgument("SymmetricKeyVariant did not unwrap successfully."));
    OP_REQUIRES_OK(op_ctx,
                   const_cast<SymmetricKeyVariant<T>*>(secret_key_var)
                       ->MaybeLazyDecode(shell_ctx_var->ct_context_,
                                         shell_ctx_var->noise_variance_));
    OP_REQUIRES(op_ctx, secret_key_var->key != nullptr,
                InvalidArgument(
                    "SymmetricKeyVariant key did not unwrap successfully."));
    Key secret_key = *secret_key_var->key;  // Deep copy.

    // Allocate a scalar output tensor to store the reduced key.
    Tensor* out;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &out));

    OP_REQUIRES_OK(op_ctx, secret_key.ModReduce());

    // Store the reduced key in the output tensor. Keep a reference to the
    // original context (even though it has the un-reduced moduli) to ensure
    // the moduli held internally by the key are not deleted prematurely.
    SymmetricKeyVariant<T> reduced_key_variant(std::move(secret_key),
                                               secret_key_var->ct_context);
    out->scalar<Variant>()() = std::move(reduced_key_variant);
  }
};

template <typename T>
class ModulusReduceCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit ModulusReduceCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);
    OP_REQUIRES(op_ctx, a.dim_size(0) > 0,
                InvalidArgument("Cannot modulus reduce an empty ciphertext."));
    auto flat_a = a.flat<Variant>();

    // Allocate the output tensor.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));
    auto flat_output = output->flat<Variant>();

    // Gather the parameters for the modulus reduction.
    size_t level = shell_ctx->NumMainPrimeModuli() - 1;
    auto q_inv_mod_qs = shell_ctx->MainPrimeModulusInverseResidues();
    OP_REQUIRES(
        op_ctx, level < q_inv_mod_qs.size(),
        InvalidArgument(
            "Ciphertext level does not match num inverse prime moduli."));
    OP_REQUIRES(op_ctx, level < q_inv_mod_qs[level].zs.size(),
                InvalidArgument("Ciphertext level does not match rns size."));
    auto ql_inv = q_inv_mod_qs[level].Prefix(level);

    auto t = shell_ctx->PlaintextModulus();
    auto main_moduli = shell_ctx->MainPrimeModuli();
    auto reduced_moduli = shell_ctx->MainPrimeModuli();
    reduced_moduli.pop_back();

    auto ct_col_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        // Get the ciphertext wrapper from the input.
        SymmetricCtVariant<T> const* ct_a_var =
            flat_a(i).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(op_ctx, ct_a_var != nullptr,
                    InvalidArgument("SymmetricCtVariant at flat index:", i,
                                    " did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt result_ct =
            ct_a_var->ct;  // Deep copy. ModReduce is in place.

        OP_REQUIRES_OK(op_ctx, result_ct.ModReduce(t, ql_inv));

        // Store in the output. Keep a reference to the original context to
        // ensure the moduli held internally by the ciphertext are not deleted
        // prematurely.
        SymmetricCtVariant<T> result_var(
            std::move(result_ct), ct_a_var->ct_context, ct_a_var->error_params);
        flat_output(i) = std::move(result_var);
      }
    };
    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_red = 618917;  // ns, measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_red,
                             ct_col_in_range);
  }
};

template <typename T>
class ModulusReducePtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit ModulusReducePtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);
    OP_REQUIRES(op_ctx, a.dim_size(0) > 0,
                InvalidArgument("Cannot modulus reduce an empty ciphertext."));
    auto flat_a = a.flat<Variant>();

    // Allocate the output tensor.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));
    auto flat_output = output->flat<Variant>();

    // Gather the parameters for the modulus reduction.
    size_t level = shell_ctx->NumMainPrimeModuli() - 1;
    auto q_inv_mod_qs = shell_ctx->MainPrimeModulusInverseResidues();
    OP_REQUIRES(
        op_ctx, level < q_inv_mod_qs.size(),
        InvalidArgument(
            "Polynomial level does not match num inverse prime moduli."));
    OP_REQUIRES(op_ctx, level < q_inv_mod_qs[level].zs.size(),
                InvalidArgument("Polynomial level does not match rns size."));
    auto ql_inv = q_inv_mod_qs[level].Prefix(level);
    auto main_moduli = shell_ctx->MainPrimeModuli();
    auto t = shell_ctx->PlaintextModulus();

    auto pt_col_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        PolynomialVariant<T> const* pt_a_var =
            flat_a(i).get<PolynomialVariant<T>>();
        OP_REQUIRES(op_ctx, pt_a_var != nullptr,
                    InvalidArgument("PolynomialVariant at flat index:", i,
                                    " did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<PolynomialVariant<T>*>(pt_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_));

        // Deep copy the polynomial because ModReduce is in place.
        OP_REQUIRES_VALUE(RnsPolynomial pt_a, op_ctx,
                          RnsPolynomial::Create(pt_a_var->poly.Coeffs(),
                                                pt_a_var->poly.IsNttForm()));

        OP_REQUIRES_OK(op_ctx, pt_a.ModReduceLsb(t, ql_inv, main_moduli));

        PolynomialVariant<T> result_var(std::move(pt_a),
                                        shell_ctx_var->ct_context_);
        flat_output(i) = std::move(result_var);
      }
    };
    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_red = 472447;  // ns, measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_red,
                             pt_col_in_range);
  }
};

REGISTER_KERNEL_BUILDER(Name("ModulusReduceContext64").Device(DEVICE_CPU),
                        ModulusReduceContextOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ModulusReduceKey64").Device(DEVICE_CPU),
                        ModulusReduceKeyOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ModulusReduceCt64").Device(DEVICE_CPU),
                        ModulusReduceCtOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("ModulusReducePt64").Device(DEVICE_CPU),
                        ModulusReducePtOp<uint64>);
