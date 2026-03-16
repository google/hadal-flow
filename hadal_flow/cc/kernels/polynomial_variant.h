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

#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "utils.h"

using tensorflow::Status;
using tensorflow::tstring;
using tensorflow::VariantTensorData;
using tensorflow::errors::InvalidArgument;

template <typename T>
class PolynomialVariant {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Polynomial = rlwe::RnsPolynomial<ModularInt>;
  std::vector<ModularInt> dummy = {static_cast<ModularInt>(0)};

 public:
  PolynomialVariant() : poly(Polynomial::Create({dummy}, false).value()) {}

  PolynomialVariant(Polynomial arg, std::shared_ptr<Context const> ct_context_)
      : poly(std::move(arg)), ct_context(ct_context_) {}

  static inline char const kTypeName[] = "ShellPolynomialVariant";

  std::string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const {
    auto async_poly_str = poly_str;  // Make sure key string is not deallocated.
    auto async_ct_context = ct_context;

    if (async_ct_context == nullptr) {
      // If the context is null, this may have been decoded but not lazy decoded
      // yet. In this case, directly encode the polynomial string.
      if (async_poly_str == nullptr) {
        std::cout << "ERROR: Polynomial not set, cannot encode." << std::endl;
        return;
      }
      data->tensors_.push_back(Tensor(*async_poly_str));
      return;
    }
    auto serialized_poly_or =
        poly.Serialize(async_ct_context->MainPrimeModuli());
    if (!serialized_poly_or.ok()) {
      std::cout << "ERROR: Failed to serialize polynomial: "
                << serialized_poly_or.status();
      return;
    }
    std::string serialized_poly;
    serialized_poly_or.value().SerializeToString(&serialized_poly);
    data->tensors_.push_back(Tensor(serialized_poly));
  };

  bool Decode(VariantTensorData const& data) {
    if (data.tensors_.size() != 1) {
      std::cout << "ERROR: Decode polynomial Expected 1 tensor, got "
                << data.tensors_.size() << "." << std::endl;
      return false;
    }

    if (poly_str != nullptr) {
      std::cout << "ERROR: Polynomial already decoded";
      return false;
    }

    poly_str = std::make_shared<std::string>(
        data.tensors_[0].scalar<tstring>()().begin(),
        data.tensors_[0].scalar<tstring>()().end());

    return true;
  };

  Status MaybeLazyDecode(std::shared_ptr<Context const> ct_context_) {
    std::lock_guard<std::mutex> lock(mutex.mutex);

    if (ct_context != nullptr) {
      return OkStatus();
    }

    rlwe::SerializedRnsPolynomial serialized_poly;
    bool ok = serialized_poly.ParseFromString(*poly_str);
    if (!ok) {
      return InvalidArgument("Failed to parse polynomial.");
    }

    TF_ASSIGN_OR_RETURN(
        poly, Polynomial::Deserialize(serialized_poly,
                                      ct_context_->MainPrimeModuli()));

    // Hold a pointer to the context for future encoding.
    ct_context = ct_context_;

    // Clear the serialized polynomial string.
    poly_str = nullptr;

    return OkStatus();
  };

  std::string DebugString() const { return "ShellPolynomialVariant"; }

  variant_mutex mutex;
  Polynomial poly;
  std::shared_ptr<std::string> poly_str;
  std::shared_ptr<Context const> ct_context;
};
