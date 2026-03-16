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

#include "shell_encryption/rns/rns_galois_key.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "utils.h"

using tensorflow::VariantTensorData;

template <typename T>
class RotationKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Gadget = rlwe::RnsGadget<ModularInt>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;

 public:
  RotationKeyVariant() {}

  static inline char const kTypeName[] = "ShellRotationKeyVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    auto async_key_strs = key_strs;  // Make sure key string is not deallocated.
    auto async_ct_context = ct_context;

    if (async_ct_context == nullptr) {
      // If the context is null, this may have been decoded but not lazy decoded
      // yet. In this case, directly encode the key strings.
      if (async_key_strs == nullptr) {
        std::cout << "ERROR: Rotation key not set, cannot encode." << std::endl;
        return;
      }
      data->tensors_.reserve(async_key_strs->size());
      for (auto const& key_str : *async_key_strs) {
        data->tensors_.push_back(Tensor(key_str));
      }
    }

    // Skip first rotation key at index 0.
    data->tensors_.reserve(keys.size() - 1);

    for (int i = 1; i < keys.size(); i++) {
      auto serialized_key_or = keys[i]->Serialize();
      if (!serialized_key_or.ok()) {
        std::cout << "ERROR: Failed to serialize rotation key: "
                  << serialized_key_or.status();
        return;
      }
      std::string serialized_key;
      serialized_key_or.value().SerializeToString(&serialized_key);
      data->tensors_.push_back(Tensor(serialized_key));
    }
  };

  bool Decode(VariantTensorData const& data) {
    if (data.tensors_.size() < 1) {
      std::cout << "ERROR: Not enough tensors to deserialize rotation key."
                << std::endl;
      return false;
    }

    if (key_strs != nullptr) {
      std::cout << "ERROR: Rotation key already decoded." << std::endl;
      return false;
    }

    size_t num_keys = data.tensors_.size();
    std::vector<std::string> building_key_strs;
    building_key_strs.reserve(num_keys);

    for (size_t i = 0; i < num_keys; ++i) {
      std::string const serialized_key(
          data.tensors_[i].scalar<tstring>()().begin(),
          data.tensors_[i].scalar<tstring>()().end());

      building_key_strs.push_back(std::move(serialized_key));
    }

    key_strs = std::make_shared<std::vector<std::string>>(
        std::move(building_key_strs));

    return true;
  };

  Status MaybeLazyDecode(std::shared_ptr<Context const> ct_context_) {
    std::lock_guard<std::mutex> lock(mutex.mutex);

    // If the keys have already been fully decoded, nothing to do.
    if (ct_context != nullptr) {
      return OkStatus();
    }

    // Re-create the gadget.
    int level = ct_context_->NumMainPrimeModuli() - 1;
    TF_ASSIGN_OR_RETURN(auto q_hats,
                        ct_context_->MainPrimeModulusComplements(level));
    TF_ASSIGN_OR_RETURN(auto q_hat_invs,
                        ct_context_->MainPrimeModulusCrtFactors(level));
    std::vector<size_t> log_bs(ct_context_->NumMainPrimeModuli(),
                               kLogGadgetBase);
    TF_ASSIGN_OR_RETURN(
        auto raw_gadget,
        Gadget::Create(ct_context_->LogN(), log_bs, q_hats, q_hat_invs,
                       ct_context_->MainPrimeModuli()));
    gadget = std::make_shared<Gadget>(std::move(raw_gadget));

    // Decode the keys.
    // The first key is skipped as it corresponds to a rotation by 0.
    keys.reserve(key_strs->size() + 1);
    keys.push_back(nullptr);

    for (auto const& key_str : *key_strs) {
      rlwe::SerializedRnsGaloisKey serialized_key;
      bool ok = serialized_key.ParseFromString(key_str);
      if (!ok) {
        return InvalidArgument("Failed to parse rotation key.");
      }

      // Using the moduli, reconstruct the key polynomial.
      TF_ASSIGN_OR_RETURN(RotationKey key, RotationKey::Deserialize(
                                               serialized_key, gadget.get(),
                                               ct_context_->MainPrimeModuli()));

      keys.push_back(std::make_shared<RotationKey>(std::move(key)));
    }

    // Hold a pointer to the context for future encoding.
    ct_context = ct_context_;

    // Clear the key strings.
    key_strs = nullptr;

    return OkStatus();
  };

  std::string DebugString() const { return "ShellRotationKeyVariant"; }

  variant_mutex mutex;
  // Each key holds a raw pointer to gadget. Use a smart pointer to the gadget
  // to help with copy semantics.
  std::shared_ptr<Gadget> gadget;
  // Rotation keys do not have default constructors, so use a shared pointer.
  std::vector<std::shared_ptr<RotationKey>> keys;
  std::shared_ptr<std::vector<std::string>> key_strs;
  std::shared_ptr<Context const> ct_context;
};

template <typename T>
class FastRotationKeyVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;

 public:
  FastRotationKeyVariant() {}

  // Create with gadget first, then create and add keys.
  FastRotationKeyVariant(std::vector<RnsPolynomial> keys,
                         std::shared_ptr<Context const> ct_context_)
      : keys(std::move(keys)), ct_context(ct_context_) {}

  static inline char const kTypeName[] = "ShellFastRotationKeyVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    auto async_key_strs = key_strs;  // Make sure key string is not deallocated.
    auto async_ct_context = ct_context;

    if (async_ct_context == nullptr) {
      // If the context is null, this may have been decoded but not lazy decoded
      // yet. In this case, directly encode the key strings.
      if (async_key_strs == nullptr) {
        std::cout << "ERROR: Fast rotation key not set, cannot encode."
                  << std::endl;
        return;
      }
      data->tensors_.reserve(async_key_strs->size());
      for (auto const& key_str : *async_key_strs) {
        data->tensors_.push_back(Tensor(key_str));
      }
    }

    data->tensors_.reserve(keys.size());

    for (auto const& key : keys) {
      auto serialized_key_or =
          key.Serialize(async_ct_context->MainPrimeModuli());
      if (!serialized_key_or.ok()) {
        std::cout << "ERROR: Failed to serialize key: "
                  << serialized_key_or.status();
        return;
      }
      std::string serialized_key;
      serialized_key_or.value().SerializeToString(&serialized_key);
      data->tensors_.push_back(Tensor(serialized_key));
    }
  };

  bool Decode(VariantTensorData const& data) {
    static bool warning_printed = false;
    if (!warning_printed) {
      std::cout
          << "WARNING: Deserializing secret fast rotation key. This should "
             "only happen on the appropriate party."
          << std::endl;
      warning_printed = true;
    }
    if (data.tensors_.size() < 1) {
      std::cout << "ERROR: Not enough tensors to deserialize fast rotation key."
                << std::endl;
      return false;
    }

    if (key_strs != nullptr) {
      std::cout << "ERROR: Fast rotation key already decoded." << std::endl;
      return false;
    }

    size_t num_keys = data.tensors_.size();
    std::vector<std::string> building_key_strs;
    building_key_strs.reserve(num_keys);

    for (size_t i = 0; i < data.tensors_.size(); ++i) {
      std::string const serialized_key(
          data.tensors_[i].scalar<tstring>()().begin(),
          data.tensors_[i].scalar<tstring>()().end());

      building_key_strs.push_back(std::move(serialized_key));
    }

    key_strs = std::make_shared<std::vector<std::string>>(
        std::move(building_key_strs));

    return true;
  };

  Status MaybeLazyDecode(std::shared_ptr<Context const> ct_context_) {
    std::lock_guard<std::mutex> lock(mutex.mutex);

    // If the keys have already been fully decoded, nothing to do.
    if (ct_context != nullptr) {
      return OkStatus();
    }

    for (auto const& key_str : *key_strs) {
      rlwe::SerializedRnsPolynomial serialized_key;
      bool ok = serialized_key.ParseFromString(key_str);
      if (!ok) {
        return InvalidArgument("Failed to parse fast rotation key polynomial.");
      }

      // Using the moduli, reconstruct the key polynomial.
      TF_ASSIGN_OR_RETURN(auto key_polynomial,
                          rlwe::RnsPolynomial<ModularInt>::Deserialize(
                              serialized_key, ct_context_->MainPrimeModuli()));

      keys.push_back(std::move(key_polynomial));
    }

    // Hold a pointer to the context for future encoding.
    ct_context = ct_context_;

    // Clear the key strings.
    key_strs = nullptr;

    return OkStatus();
  };

  std::string DebugString() const { return "ShellFastRotationKeyVariant"; }

  variant_mutex mutex;
  std::vector<RnsPolynomial> keys;
  std::shared_ptr<std::vector<std::string>> key_strs;
  std::shared_ptr<Context const> ct_context;
};