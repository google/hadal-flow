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

#include <type_traits>

#include "context_variant.h"
#include "polynomial_variant.h"
#include "shell_encryption/context.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using rlwe::SecurePrng;
using rlwe::Uint64;
using tensorflow::DEVICE_CPU;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::int8;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::uint8;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

template <typename From, typename To>
class PolynomialImportOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<To>;
  using Integer = typename ModularInt::Int;
  using ModularIntParams = typename ModularInt::Params;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using NttParams = rlwe::NttParameters<ModularInt>;

  float scaling_factor_ = 1;
  bool random_round_ = false;

 public:
  explicit PolynomialImportOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("scaling_factor", &scaling_factor_));
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("random_round", &random_round_));

    if constexpr (!std::is_floating_point<From>::value) {
      OP_REQUIRES(op_ctx, scaling_factor_ == 1.,
                  InvalidArgument("scaling_factor must be 1 when using integer "
                                  "(non-floating) type. Saw scaling_factor: ",
                                  scaling_factor_));
    }
    OP_REQUIRES(op_ctx, scaling_factor_ > 0.,
                InvalidArgument("scaling_factor must be positive."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<To> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<To>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    Encoder const* encoder = shell_ctx_var->encoder_.get();

    Tensor const& input = op_ctx->input(1);

    int num_slots = 1 << shell_ctx->LogN();

    // Allocate output tensor with the same shape as the input tensor except
    // for the first dimension which is removed since it is a polynomial
    // where each slot is packed by the first dimension of the tensor.
    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.RemoveDimWithStatus(0));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    // Set up flat views of the input and output tensors.
    auto flat_input = input.flat_outer_dims<From>();
    auto flat_output = output->flat<Variant>();

    auto import_in_range = [&](int start, int end, int worker_id) {
      for (int i = start; i < end; ++i) {
        // Contiguous memory of plaintext for the absl span required by RNS BGV
        // encoder.
        std::vector<To> wrapped_nums;

        // Ensure input dimension doesn't exceed num_slots
        int input_slots = std::min(static_cast<int>(flat_input.dimension(0)),
                                   static_cast<int>(num_slots));

        if constexpr (std::is_signed<From>::value) {
          // SHELL is built on the assumption that the plaintext type (in this
          // case `From`) will always fit into the ciphertext underlying type
          // (in this case `To`). I.e. the plaintext modulus is stored as the
          // ciphertext type. This is true even in the RNS code paths. This
          // means that this function can convert `From` to a signed version of
          // `To`, then modulus switch into plaintext field t and type `To`
          // without overflow.
          using SignedInteger = std::make_signed_t<To>;

          // Contiguous memory for signed `To`'s to be encoded to unsigned data
          // type.
          std::vector<SignedInteger> nums(num_slots, 0);  // Initialize to zeros

          if constexpr (std::is_floating_point<From>::value) {
            // Floating point requires rounding. This can be done randomly or
            // deterministically.
            if (random_round_) {
              // Use a separate PRNG for each thread to avoid contention.
              int prng_i = worker_id % shell_ctx_var->prng_.size();
              SecurePrng* prng = shell_ctx_var->prng_[prng_i].get();

              for (int slot = 0; slot < input_slots; ++slot) {
                double val =
                    static_cast<double>(flat_input(slot, i)) * scaling_factor_;
                OP_REQUIRES_VALUE(nums[slot], op_ctx,
                                  RandomRound<SignedInteger>(prng, val));
              }
            } else {
              for (int slot = 0; slot < input_slots; ++slot) {
                double val =
                    static_cast<double>(flat_input(slot, i)) * scaling_factor_;
                nums[slot] = static_cast<SignedInteger>(round(val));
              }
            }
          } else {
            // If From is signed integer, just cast and copy.
            for (int slot = 0; slot < input_slots; ++slot) {
              nums[slot] = static_cast<SignedInteger>(flat_input(slot, i));
            }
          }

          // Map signed integers into the plaintext modulus field.
          OP_REQUIRES_VALUE(
              wrapped_nums, op_ctx,
              (encoder->template WrapSigned<SignedInteger>(nums)));
        } else {
          wrapped_nums = std::vector<To>(num_slots, 0);  // Initialize to zeros
          // Since From and To are both unsigned, no need to encode signed
          // values into unsigned type. Just cast and copy.
          for (int slot = 0; slot < input_slots; ++slot) {
            wrapped_nums[slot] = static_cast<To>(flat_input(slot, i));
          }
        }

        // The encoder first performs an inverse ntt (mod t), then switches to
        // to mod Q in RNS form. This is important so that subsequent operations
        // on the polynomial happen element-wise in the plaintext space.
        // Note "importing" the integers in the correct modulus (first t, then
        // switching to Q) is non-trivial when plaintext numbers are negative.
        OP_REQUIRES_VALUE(
            RnsPolynomial rns_polynomial, op_ctx,
            encoder->EncodeBgv(wrapped_nums, shell_ctx->MainPrimeModuli()));

        // Wrap in a PolynomialVariant and store in output tensor.
        auto variant = PolynomialVariant<To>(std::move(rns_polynomial),
                                             shell_ctx_var->ct_context_);
        flat_output(i) = std::move(variant);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_import = 70 * num_slots;  // ns, measured on log_n = 11
    thread_pool->ParallelForWithWorkerId(
        flat_input.dimension(1),
        tsl::thread::ThreadPool::SchedulingParams::Adaptive(cost_per_import),
        import_in_range);
  }
};

template <typename From, typename To>
class PolynomialExportOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<From>;
  using ModularIntParams = typename ModularInt::Params;
  using Polynomial = rlwe::Polynomial<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using NttParams = rlwe::NttParameters<ModularInt>;

  float scaling_factor_ = 1;

 public:
  explicit PolynomialExportOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("scaling_factor", &scaling_factor_));

    if constexpr (!std::is_floating_point<To>::value) {
      OP_REQUIRES(
          op_ctx, scaling_factor_ == 1.,
          InvalidArgument("scaling_factor must be 1. when using integer "
                          "(non-floating) type. Saw scaling_factor: ",
                          scaling_factor_));
    }
    OP_REQUIRES(op_ctx, scaling_factor_ > 0.,
                InvalidArgument("scaling_factor must be positive."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<From> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<From>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    Encoder const* encoder = shell_ctx_var->encoder_.get();

    Tensor const& input = op_ctx->input(1);

    size_t num_slots = 1 << shell_ctx->LogN();

    // Allocate the output tensor including the first dimension which is the
    // slots of the polynomial after unpacking.
    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.InsertDimWithStatus(0, num_slots));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_input = input.flat<Variant>();
    auto flat_output = output->flat_outer_dims<To>();

    auto export_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        PolynomialVariant<From> const* pv =
            flat_input(i).get<PolynomialVariant<From>>();
        OP_REQUIRES(op_ctx, pv != nullptr,
                    InvalidArgument("PolynomialVariant at flat index: ", i,
                                    " did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx, const_cast<PolynomialVariant<From>*>(pv)->MaybeLazyDecode(
                        shell_ctx_var->ct_context_));

        // Deep copy the polynomial.
        OP_REQUIRES_VALUE(
            RnsPolynomial rns_polynomial, op_ctx,
            RnsPolynomial::Create(pv->poly.Coeffs(), pv->poly.IsNttForm()));

        // TODO: if debug
        OP_REQUIRES(op_ctx,
                    rns_polynomial.NumCoeffs() == static_cast<int>(num_slots),
                    InvalidArgument(
                        "Polynomial dimensions: ", rns_polynomial.NumCoeffs(),
                        " do not match shell context degree: ", num_slots));

        // TODO: if debug
        OP_REQUIRES(op_ctx, rns_polynomial.IsNttForm(),
                    InvalidArgument("PolynomialVariant at flat index: ", i,
                                    " is not in NTT form."));

        // Switch from mod Q to plaintext modulus and compute NTT to get the
        // plaintext by using the Decode function.
        OP_REQUIRES_VALUE(
            std::vector<From> unsigned_nums, op_ctx,
            encoder->DecodeBgv(rns_polynomial, shell_ctx->MainPrimeModuli()));

        if constexpr (std::is_signed<To>::value) {
          // Map the plaintext modulus field back into signed integers.
          // Effectively switches the modulus from t to 2^(num bits in `From`)
          // handling the sign bits appropriately.
          OP_REQUIRES_VALUE(
              std::vector<std::make_signed_t<From>> nums, op_ctx,
              encoder->template UnwrapToSigned<std::make_signed_t<From>>(
                  unsigned_nums));

          for (size_t slot = 0; slot < num_slots; ++slot) {
            To to_val = static_cast<To>(nums[slot]);
            if constexpr (std::is_floating_point<To>::value) {
              to_val /= scaling_factor_;
            }
            flat_output(slot, i) = to_val;
          }
        } else {
          // both `From` and `To` are unsigned, just cast and copy.
          for (size_t slot = 0; slot < num_slots; ++slot) {
            flat_output(slot, i) = static_cast<To>(unsigned_nums[slot]);
          }
        }
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_export = 70 * num_slots;  // ns, measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(1), cost_per_export,
                             export_in_range);
  }
};

// Import ops.
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint8>("Dtype"),
                        PolynomialImportOp<uint8, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("PolynomialImport64").Device(DEVICE_CPU).TypeConstraint<int8>("Dtype"),
    PolynomialImportOp<int8, uint64>);

REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint16>("Dtype"),
                        PolynomialImportOp<uint16, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int16>("Dtype"),
                        PolynomialImportOp<int16, uint64>);

REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint32>("Dtype"),
                        PolynomialImportOp<uint32, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("Dtype"),
                        PolynomialImportOp<int32, uint64>);

REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint64>("Dtype"),
                        PolynomialImportOp<uint64, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Dtype"),
                        PolynomialImportOp<int64, uint64>);

REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        PolynomialImportOp<float, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialImport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        PolynomialImportOp<double, uint64>);

// Export ops.
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint8>("dtype"),
                        PolynomialExportOp<uint64, uint8>);
REGISTER_KERNEL_BUILDER(
    Name("PolynomialExport64").Device(DEVICE_CPU).TypeConstraint<int8>("dtype"),
    PolynomialExportOp<uint64, int8>);

REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint16>("dtype"),
                        PolynomialExportOp<uint64, uint16>);
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int16>("dtype"),
                        PolynomialExportOp<uint64, int16>);

REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint32>("dtype"),
                        PolynomialExportOp<uint64, uint32>);
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("dtype"),
                        PolynomialExportOp<uint64, int32>);

REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint64>("dtype"),
                        PolynomialExportOp<uint64, uint64>);
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("dtype"),
                        PolynomialExportOp<uint64, int64>);

REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("dtype"),
                        PolynomialExportOp<uint64, float>);
REGISTER_KERNEL_BUILDER(Name("PolynomialExport64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("dtype"),
                        PolynomialExportOp<uint64, double>);

typedef PolynomialVariant<uint64> PolynomialVariantUint64;
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(PolynomialVariantUint64,
                                       PolynomialVariantUint64::kTypeName);