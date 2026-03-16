/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <memory>

#include "shell_encryption/prng/hkdf_prng.h"
#include "shell_encryption/rns/coefficient_encoder.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_error_params.h"
#include "shell_encryption/rns/rns_gadget.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "shell_encryption/rns/rns_secret_key.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "utils.h"

using tensorflow::DT_UINT64;
using tensorflow::Status;
using tensorflow::tstring;
using tensorflow::VariantTensorData;

// This class wraps SHELL encryption library objects to store state required for
// performing homomorphic operations on encrypted data.
//
// Tensorflow requires copy constructors be defined while many of objects in
// this class from the SHELL encryption library have their copy constructors
// deleted to prevent users from performance issues by accidental copies.
// As such, the copy constructors defined here need to create copies of the
// shell objects without calling the copy constructors.
template <typename T>
class ContextVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ModularIntParams = typename rlwe::MontgomeryInt<T>::Params;
  using NttParams = rlwe::NttParameters<ModularInt>;
  using Context = rlwe::RnsContext<ModularInt>;
  using ErrorParams = rlwe::RnsErrorParams<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using Gadget = rlwe::RnsGadget<ModularInt>;
  using Prng = rlwe::SecurePrng;
  using HkdfPrng = rlwe::HkdfPrng;

 public:
  ContextVariant() = default;

  absl::Status Initialize(size_t log_n, std::vector<T> qs, std::vector<T> ps,
                          T pt_modulus, size_t noise_variance,
                          std::string seed = {}) {
    // Store miminum necessary information for encode.
    log_n_ = log_n;
    qs_ = qs;
    ps_ = ps;
    pt_modulus_ = pt_modulus;
    noise_variance_ = noise_variance;
    seed_ = seed;

    // Create plaintext context objects
    HADAL_ASSIGN_OR_RETURN(pt_params_, ModularIntParams::Create(pt_modulus));
    HADAL_ASSIGN_OR_RETURN(
        auto pt_ntt_params,
        rlwe::InitializeNttParameters<ModularInt>(log_n, pt_params_.get()));
    pt_ntt_params_ =
        std::make_shared<NttParams const>(std::move(pt_ntt_params));

    // Create ciphertext context objects.
    HADAL_ASSIGN_OR_RETURN(auto ct_context,
                           Context::CreateForBgvFiniteFieldEncoding(
                               log_n_, qs_, ps_, pt_modulus_));
    ct_context_ = std::make_shared<Context const>(std::move(ct_context));

    HADAL_ASSIGN_OR_RETURN(
        auto error_params,
        ErrorParams::Create(log_n, ct_context_->MainPrimeModuli(),
                            ct_context_->AuxPrimeModuli(), BitWidth(pt_modulus),
                            sqrt(noise_variance)));
    error_params_ =
        std::make_shared<ErrorParams const>(std::move(error_params));

    HADAL_ASSIGN_OR_RETURN(auto encoder, Encoder::Create(ct_context_.get()));
    encoder_ = std::make_shared<Encoder const>(std::move(encoder));

    // Create the gadget.
    int level = qs_.size() - 1;
    HADAL_ASSIGN_OR_RETURN(auto q_hats,
                           ct_context_->MainPrimeModulusComplements(level));
    HADAL_ASSIGN_OR_RETURN(auto q_hat_invs,
                           ct_context_->MainPrimeModulusCrtFactors(level));
    std::vector<size_t> log_bs(qs_.size(), ContextVariant<T>::kLogGadgetBase);
    HADAL_ASSIGN_OR_RETURN(auto gadget,
                           Gadget::Create(log_n_, log_bs, q_hats, q_hat_invs,
                                          ct_context_->MainPrimeModuli()));
    gadget_ = std::make_shared<Gadget const>(std::move(gadget));

    // If the PRNG seed is empty, generate one.
    std::string prng_seed;
    if (seed.empty()) {
      HADAL_ASSIGN_OR_RETURN(prng_seed, HkdfPrng::GenerateSeed());
    } else {
      prng_seed = seed;
    }

    // Create the PRNGs, one per core.
    auto processor_count = std::thread::hardware_concurrency();
    if (processor_count == 0) {
      std::cout << "WARNING: Failed to detect the number of processors. "
                << std::endl;
      processor_count = 1;
    }
    prng_.reserve(processor_count);

    // Create the first prng then seed the rest with its own output. Note that
    // if a seed is given, all PRNGs cannot use the same seed, but their output
    // should still be deterministic.
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Prng> ptr, HkdfPrng::Create(prng_seed));
    prng_.push_back(std::move(ptr));
    for (size_t i = 1; i < processor_count; ++i) {
      std::vector<uint8_t> gen_rand;
      gen_rand.reserve(64);
      for (int i = 0; i < 64; ++i) {
        TF_ASSIGN_OR_RETURN(uint8_t rand, prng_[0]->Rand8());
        gen_rand.push_back(rand);
      }
      std::string gen_seed(gen_rand.begin(), gen_rand.end());
      TF_ASSIGN_OR_RETURN(ptr, HkdfPrng::Create(gen_seed));
      prng_.push_back(std::move(ptr));
    }

    // Initialize the substitution powers used for rotation, namely
    // the rotation base power (e.g. 5^i mod 2n).
    uint num_slots = 1 << log_n_;
    uint two_n = num_slots << 1;
    uint sub_power = 1;
    substitution_powers_.reserve(num_slots / 2);
    for (uint shift = 0; shift < num_slots / 2; ++shift) {
      substitution_powers_.push_back(sub_power);
      sub_power *= kSubstitutionBasePower;
      sub_power %= two_n;
    }

    return absl::OkStatus();
  }

  static inline char const kTypeName[] = "ShellContextVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    if constexpr (std::is_same<T, uint64_t>::value) {
      Tensor log_n_tensor = Tensor(log_n_);
      Tensor qs_tensor = Tensor(DT_UINT64, TensorShape({int64_t(qs_.size())}));
      for (size_t i = 0; i < qs_.size(); ++i) {
        qs_tensor.flat<uint64_t>()(i) = qs_[i];
      }
      Tensor ps_tensor = Tensor(DT_UINT64, TensorShape({int64_t(ps_.size())}));
      for (size_t i = 0; i < ps_.size(); ++i) {
        ps_tensor.flat<uint64_t>()(i) = ps_[i];
      }
      Tensor pt_modulus_tensor = Tensor(pt_modulus_);
      Tensor noise_variance_tensor = Tensor(noise_variance_);
      Tensor seed_tensor = Tensor(seed_);
      Tensor substitution_powers_tensor = Tensor(
          DT_UINT64, TensorShape({int64_t(substitution_powers_.size())}));
      for (size_t i = 0; i < substitution_powers_.size(); ++i) {
        substitution_powers_tensor.flat<uint64_t>()(i) =
            substitution_powers_[i];
      }

      data->tensors_.reserve(7);
      data->tensors_.push_back(log_n_tensor);
      data->tensors_.push_back(qs_tensor);
      data->tensors_.push_back(ps_tensor);
      data->tensors_.push_back(pt_modulus_tensor);
      data->tensors_.push_back(noise_variance_tensor);
      data->tensors_.push_back(seed_tensor);
      data->tensors_.push_back(substitution_powers_tensor);
    }
  };

  bool Decode(VariantTensorData const& data) {
    if (data.tensors_.size() != 7) {
      return false;
    }

    size_t log_n = data.tensors_[0].scalar<size_t>()(0);

    std::vector<T> qs;
    qs.reserve(data.tensors_[1].NumElements());
    for (int64_t i = 0; i < data.tensors_[1].NumElements(); ++i) {
      qs.push_back(data.tensors_[1].vec<uint64_t>()(i));
    }

    std::vector<T> ps;
    ps.reserve(data.tensors_[2].NumElements());
    for (int64_t i = 0; i < data.tensors_[2].NumElements(); ++i) {
      ps.push_back(data.tensors_[2].vec<uint64_t>()(i));
    }

    T pt_modulus = data.tensors_[3].scalar<T>()(0);

    size_t noise_variance = data.tensors_[4].scalar<size_t>()(0);

    tstring seed = data.tensors_[5].scalar<tstring>()(0);
    std::string std_seed(seed.data(), seed.size());

    std::vector<uint> substitution_powers;
    substitution_powers.reserve(data.tensors_[6].NumElements());
    for (int64_t i = 0; i < data.tensors_[6].NumElements(); ++i) {
      substitution_powers.push_back(data.tensors_[6].vec<uint64_t>()(i));
    }

    Status s = Initialize(log_n, qs, ps, pt_modulus, noise_variance, std_seed);
    return s.ok();
  };

  std::string DebugString() const { return "ShellContextVariant"; }

  static constexpr int kLogGadgetBase = 4;

  size_t log_n_;
  std::vector<T> qs_;
  std::vector<T> ps_;
  T pt_modulus_;
  size_t noise_variance_;
  std::string seed_;
  std::vector<uint> substitution_powers_;

  // Ideally these members wouldn't be smart pointers (plain pointers or even
  // just the objects), but many of them don't have default constructors and
  // in some places SHELL expects callers to use smart pointers.
  // Futhermore, other objects in hadal hold shared_ptrs to these objects,
  // because sometimes tensorflow decides to delete this ContextVariant
  // while the moduli held in these objects are still needed for encoding.
  std::shared_ptr<ModularIntParams const> pt_params_;
  std::shared_ptr<NttParams const> pt_ntt_params_;
  std::shared_ptr<Context const> ct_context_;
  std::shared_ptr<ErrorParams const> error_params_;
  std::shared_ptr<Encoder const> encoder_;
  std::shared_ptr<Gadget const> gadget_;
  std::vector<std::shared_ptr<Prng>> prng_;
};
