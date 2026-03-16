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
#include "rotation_variants.h"
#include "shell_encryption/context.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_bgv_ciphertext.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "symmetric_variants.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using tensorflow::DEVICE_CPU;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::int8;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::uint8;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

constexpr rlwe::PrngType kPrngType = rlwe::PRNG_TYPE_HKDF;

// Fast Rotation Kernels:
// These kernels perform a faster method for reduce_sum on BGV/BFV style
// ciphertexts, skipping the keyswitching step. The reduce_summed ciphertext is
// no longer valid under the original secret key, however the plaintext can
// still be recovered through a specialized decryption process. This means a
// fast_reduce_summed ciphertext has limited subsequent operations, i.e. only
// add / multiply by plaintexts are supported.
//
// Say a degree one ciphertext is defined as c=(b, a) where b = -a * s + m + e.
// Usually, reduce sum uses key switching such that d = reduce_sum(c) decrypts
// using s, by computing b + a * s.
// Instead, we can compute d' = fast_reduce_sum(c) = (sum_i b(X^5^i) for i in
// [0, n/2], a). Note d' is not a valid ciphertext under s in that the usual
// decryption function does not output m + e. To recover the message m, we can
// compute m + e = sum_i b(X^5^i) + sum_i a(X^5^i) * s(X^5^i).

// Generate "fast rotation keys", used to decrypt a ciphertext which has been
// "fast reduced_summed". The keys consist of the secret key evaluated at
// various powers of 5, i.e. if the secret key is s(X), the keys are s(X),
// s(X^5), s(X^25), s(X^125), etc.
template <typename T>
class FastRotationKeyGenOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit FastRotationKeyGenOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    std::cout << "INFO: Generating fast rotation key" << std::endl;
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    OP_REQUIRES(op_ctx, shell_ctx_var != nullptr,
                InvalidArgument("ContextVariant did not unwrap successfully."));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    OP_REQUIRES_VALUE(SymmetricKeyVariant<T> const* secret_key_var, op_ctx,
                      GetVariant<SymmetricKeyVariant<T>>(op_ctx, 1));
    OP_REQUIRES(
        op_ctx, secret_key_var != nullptr,
        InvalidArgument("SymmetricKeyVariant did not unwrap successfully."));
    OP_REQUIRES_OK(op_ctx,
                   const_cast<SymmetricKeyVariant<T>*>(secret_key_var)
                       ->MaybeLazyDecode(shell_ctx_var->ct_context_,
                                         shell_ctx_var->noise_variance_));
    std::shared_ptr<Key> const secret_key = secret_key_var->key;

    // Allocate the output tensor which is a scalar containing the fast rotation
    // key.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, TensorShape{}, &output));

    // This method of rotation only allows us to rotate within half of the
    // polynomial slots. E.g. for n slots, slot 0 can be rotated to at most
    // n/2-1 and n/2 to n-1. This has implications for how batching is done if
    // performing back propagation under encryption.
    uint num_slots = 1 << shell_ctx->LogN();

    std::vector<RnsPolynomial> keys;
    keys.reserve(num_slots / 2);

    RnsPolynomial key_sub_i = secret_key->Key();
    keys.push_back(key_sub_i);

    for (uint i = 1; i < num_slots / 2; ++i) {
      OP_REQUIRES_VALUE(key_sub_i, op_ctx,
                        key_sub_i.Substitute(kSubstitutionBasePower,
                                             shell_ctx->MainPrimeModuli()));
      keys.push_back(key_sub_i);
    }

    // Deep copy the prime moduli for storage in the FastRotationKeyVariant.
    std::vector<rlwe::PrimeModulus<ModularInt> const*> prime_moduli_copy;
    prime_moduli_copy.reserve(shell_ctx->MainPrimeModuli().size());
    for (auto const& modulus : shell_ctx->MainPrimeModuli()) {
      prime_moduli_copy.push_back(modulus);
    }

    FastRotationKeyVariant key_var(std::move(keys), shell_ctx_var->ct_context_);
    output->flat<Variant>()(0) = std::move(key_var);
  }
};

// This is a faster version of the ReduceSumByRotationOp that does not use
// keyswitching. The output is not a valid ciphertext under the original
// secret key, but can still be decrypted.
//
// If a degree 1 input ciphertext is the tuple (b, a) where b = a*s + m + e,
// this operation computes the sum_i b(X^5^i) for i = 0 .. n/2 - 1. The output
// is a ciphertext (sum_i b(X^5^i), a).
template <typename T>
class FastReduceSumByRotationOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using Modulus = rlwe::PrimeModulus<ModularInt>;

 public:
  explicit FastReduceSumByRotationOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    OP_REQUIRES(op_ctx, shell_ctx_var != nullptr,
                InvalidArgument("ContextVariant did not unwrap successfully."));
    auto const& sub_powers = shell_ctx_var->substitution_powers_;

    // Recover the input tensor.
    Tensor const& value = op_ctx->input(1);
    OP_REQUIRES(op_ctx, value.dim_size(0) > 0,
                InvalidArgument("Cannot fast_reduce_sum an empty ciphertext."));
    auto flat_value = value.flat<Variant>();

    // Recover num_slots and moduli from first ciphertext.
    SymmetricCtVariant<T> const* ct_var =
        flat_value(0).get<SymmetricCtVariant<T>>();
    OP_REQUIRES(
        op_ctx, ct_var != nullptr,
        InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
    OP_REQUIRES_OK(
        op_ctx, const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
    SymmetricCt const& ct = ct_var->ct;
    int num_slots = 1 << ct.LogN();
    int num_components = ct.NumModuli();
    absl::Span<Modulus const* const> moduli = ct.Moduli();
    std::vector<Modulus const*> moduli_vector;
    moduli_vector.assign(moduli.begin(), moduli.end());

    // Allocate the output tensor which is the same size as the input tensor,
    // TensorFlow's reduce_sum has slightly different semantics than this
    // operation. This operation affects top and bottom halves independently, as
    // well as repeating the sum across the halves such that the output is
    // the same shape as the input.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, value.shape(), &output));
    auto flat_output = output->flat<Variant>();

    auto reduce_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        // Learn how many slots there are from first ciphertext and create a
        // deep copy to hold the sum.
        SymmetricCtVariant<T> const* ct_var =
            flat_value(i).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(op_ctx, ct_var != nullptr,
                    InvalidArgument(
                        "SymmetricCtVariant a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        OP_REQUIRES(op_ctx, ct_var->ct.Degree() == 1,
                    InvalidArgument("Only Degree 1 ciphertexts supported."));
        SymmetricCt const& ct = ct_var->ct;

        // Add the rotations to the sum. Note the ciphertext rotations operate
        // on each half of the ciphertext separately. So the max rotation is by
        // half the number of slots.
        OP_REQUIRES_VALUE(RnsPolynomial sum_component_zero, op_ctx,
                          ct.Component(0));  // deep copy to start the sum.

        for (uint shift = 1; shift < uint(num_slots / 2); shift <<= 1) {
          // Rotate by the shift.
          OP_REQUIRES_VALUE(
              RnsPolynomial sum_shifted, op_ctx,
              sum_component_zero.Substitute(sub_powers[shift], moduli));

          // Add to the sum.
          OP_REQUIRES_OK(op_ctx,
                         sum_component_zero.AddInPlace(sum_shifted, moduli));
        }

        // The second component of the ciphertext is unchanged.
        OP_REQUIRES_VALUE(RnsPolynomial passthrough_component_one, op_ctx,
                          ct.Component(1));

        std::vector<RnsPolynomial> components{
            std::move(sum_component_zero),
            std::move(passthrough_component_one),
        };

        SymmetricCt ct_out(std::move(components), moduli_vector, ct.PowerOfS(),
                           ct.Error() * ct.LogN(), ct.ErrorParams());
        SymmetricCtVariant ct_out_var(std::move(ct_out), ct_var->ct_context,
                                      ct_var->error_params);
        flat_output(i) = std::move(ct_out_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_reduce =
        10 * num_slots * num_components;  // ns measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_reduce,
                             reduce_in_range);
  }
};

// DecryptFastRotatedOp decrypts ciphertexts which have been
// "fast_reduced_summed".
// Say a degree one ciphertext is defined as (b, a) where b = -a * s + m + e.
// Usually decryption is done by computing b + a * s. Instead, this op takes as
// input (sum_i b(X^5^i), a), and computes sum_i (b(X^5^i)) + sum_i (a(X^5^i) *
// s(X^5^i)) to decrypt.
template <typename From, typename To>
class DecryptFastRotatedOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<From>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using Key = rlwe::RnsRlweSecretKey<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

  float scaling_factor_ = 1.;

 public:
  explicit DecryptFastRotatedOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("scaling_factor", &scaling_factor_));

    if constexpr (!std::is_floating_point<To>::value) {
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
    OP_REQUIRES_VALUE(ContextVariant<From> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<From>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    auto moduli = shell_ctx->MainPrimeModuli();
    Encoder const* encoder = shell_ctx_var->encoder_.get();
    auto const& sub_powers = shell_ctx_var->substitution_powers_;

    OP_REQUIRES_VALUE(FastRotationKeyVariant<From> const* key_var, op_ctx,
                      GetVariant<FastRotationKeyVariant<From>>(op_ctx, 1));
    OP_REQUIRES(op_ctx, key_var != nullptr,
                InvalidArgument("Failed to unwrap FastRotationKeyVariant."));
    OP_REQUIRES_OK(op_ctx, const_cast<FastRotationKeyVariant<From>*>(key_var)
                               ->MaybeLazyDecode(shell_ctx_var->ct_context_));
    std::vector<RnsPolynomial> const& keys = key_var->keys;

    Tensor const& input = op_ctx->input(2);
    OP_REQUIRES(op_ctx, input.dim_size(0) > 0,
                InvalidArgument("Cannot decrypt empty ciphertext"));
    auto flat_input = input.flat<Variant>();

    // This method for decryption only works on degree 1 ciphertexts. Check the
    // first ciphertext and assume the rest are the same.
    SymmetricCtVariant<From> const* ct_var =
        flat_input(0).get<SymmetricCtVariant<From>>();
    OP_REQUIRES(
        op_ctx, ct_var != nullptr,
        InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
    OP_REQUIRES_OK(
        op_ctx, const_cast<SymmetricCtVariant<From>*>(ct_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
    OP_REQUIRES(op_ctx, ct_var->ct.Degree() == 1,
                InvalidArgument("Ciphertext must have degree 1."));

    size_t num_slots = 1 << shell_ctx->LogN();

    // Allocate the output tensor so that the shape include an extra dimension
    // at the beginning to hold values in the slots unpacked from the
    // ciphertext.
    Tensor* output;
    auto output_shape = input.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.InsertDimWithStatus(0, num_slots));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_output = output->flat_outer_dims<To>();

    auto dec_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<From> const* ct_var =
            flat_input(i).get<SymmetricCtVariant<From>>();
        OP_REQUIRES(op_ctx, ct_var != nullptr,
                    InvalidArgument("SymmetricCtVariant at flat index: ", i,
                                    " did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<From>*>(ct_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct = ct_var->ct;

        OP_REQUIRES_VALUE(RnsPolynomial ct_a, op_ctx, ct.Component(1));
        OP_REQUIRES_VALUE(RnsPolynomial ct_offset_sum, op_ctx,
                          ct_a.Mul(keys[0], moduli));

        // Compute a(X^5^i) * s(X^5^i) for i = 0 .. n/2 - 1 where a is the
        // second component of the ciphertext and s is the original secret key.
        for (uint shift = 1; shift < num_slots / 2; shift <<= 1) {
          OP_REQUIRES_VALUE(
              RnsPolynomial ct_a_sub_i, op_ctx,
              ct_offset_sum.Substitute(sub_powers[shift], moduli));

          OP_REQUIRES_OK(op_ctx, ct_offset_sum.AddInPlace(ct_a_sub_i, moduli));
        }

        // Compute b + ct_offset_sum to decrypt where b is the first component
        // of the ciphertext.
        OP_REQUIRES_VALUE(RnsPolynomial ct_b, op_ctx, ct.Component(0));
        OP_REQUIRES_VALUE(RnsPolynomial pt, op_ctx,
                          ct_b.Add(ct_offset_sum, moduli));

        // Decode() returns coefficients in underlying (e.g. uint64) form after
        // doing the outer modulo and inverse NTT.
        OP_REQUIRES_VALUE(std::vector<From> decryptions, op_ctx,
                          encoder->DecodeBgv(pt, moduli));

        if constexpr (std::is_signed<To>::value) {
          // Map the plaintext modulus field back into signed integers.
          // Effectively switches the modulus from t to 2^(num bits in `From`)
          // handling the sign bits appropriately.
          OP_REQUIRES_VALUE(
              std::vector<std::make_signed_t<From>> nums, op_ctx,
              encoder->template UnwrapToSigned<std::make_signed_t<From>>(
                  decryptions));

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
            flat_output(slot, i) = static_cast<To>(decryptions[slot]);
          }
        }
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_dec = 75 * num_slots;  // ns, measured on log_n = 11
    thread_pool->ParallelFor(flat_output.dimension(1), cost_per_dec,
                             dec_in_range);
  }
};

REGISTER_KERNEL_BUILDER(Name("FastRotationKeyGen64").Device(DEVICE_CPU),
                        FastRotationKeyGenOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("FastReduceSumByRotation64").Device(DEVICE_CPU),
                        FastReduceSumByRotationOp<uint64>);

REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint8>("dtype"),
                        DecryptFastRotatedOp<uint64, uint8>);
REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int8>("dtype"),
                        DecryptFastRotatedOp<uint64, int8>);

REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint16>("dtype"),
                        DecryptFastRotatedOp<uint64, uint16>);
REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int16>("dtype"),
                        DecryptFastRotatedOp<uint64, int16>);

REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint32>("dtype"),
                        DecryptFastRotatedOp<uint64, uint32>);
REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("dtype"),
                        DecryptFastRotatedOp<uint64, int32>);

REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<uint64>("dtype"),
                        DecryptFastRotatedOp<uint64, uint64>);
REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("dtype"),
                        DecryptFastRotatedOp<uint64, int64>);

REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("dtype"),
                        DecryptFastRotatedOp<uint64, float>);
REGISTER_KERNEL_BUILDER(Name("DecryptFastRotated64")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("dtype"),
                        DecryptFastRotatedOp<uint64, double>);

typedef FastRotationKeyVariant<uint64> FastRotationKeyVariantUint64;
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(FastRotationKeyVariantUint64,
                                       FastRotationKeyVariantUint64::kTypeName);