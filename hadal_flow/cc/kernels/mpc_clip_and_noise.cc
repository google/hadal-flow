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

#include "emp-sh2pc/emp-sh2pc.h"
#include "emp-tool/emp-tool.h"
#include "emp-tool/io/net_io_channel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using tensorflow::DEVICE_CPU;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

int const FEATURES_PARTY = emp::BOB;  // Bob is the evaluator.
int const LABELS_PARTY = emp::ALICE;  // Alice is the garbler.
constexpr bool const debug = false;

template <typename T, int Bitwidth, int Party>
void ClipAndNoise(int grad_size, T const* masks, T const* masked_grads,
                  T clipping_threshold, T const* noises,
                  T* features_party_results) {
  Integer emp_clipping_threshold(Bitwidth, clipping_threshold, LABELS_PARTY);

  Integer two_norm = Integer(Bitwidth, 0, PUBLIC);

  // First unmask the gradient and calculate the L2-norm squared.
  std::vector<Integer> grads;
  grads.reserve(grad_size);
  for (int i = 0; i < grad_size; ++i) {
    Integer mask(Bitwidth, &masks[i], FEATURES_PARTY);
    Integer masked_grad(Bitwidth, &masked_grads[i], LABELS_PARTY);

    // Unmask the gradient.
    grads.emplace_back(masked_grad - mask);

    // Calculate the L2-norm squared.
    two_norm = two_norm + (grads[i] * grads[i]);
  }

  for (int i = 0; i < grad_size; ++i) {
    Integer noise(Bitwidth, &noises[i], LABELS_PARTY);

    // Clip the gradient.
    Bit choose = two_norm.geq(emp_clipping_threshold);
    // EMP NOTE: new value = falseValue.If(bool, trueValue)
    Integer grad_or_threshold = grads[i].If(choose, emp_clipping_threshold);

    // Add noise.
    Integer noised_grad = grad_or_threshold + noise;
    T res = noised_grad.reveal<T>(FEATURES_PARTY);
    if constexpr (Party == FEATURES_PARTY) {
      // Sign extend manually.
      res = (res << (sizeof(T) * 8 - Bitwidth)) >> (sizeof(T) * 8 - Bitwidth);
      features_party_results[i] = res;
    }

    if constexpr (debug && Party == FEATURES_PARTY) {
      std::cout << "ClipAndNoise" << std::endl;
      std::cout << " Grad: " << res << std::endl;
      std::cout << " Ct: " << emp_clipping_threshold.reveal<T>(FEATURES_PARTY)
                << std::endl;
      std::cout << " Choose: " << choose.reveal<bool>(FEATURES_PARTY)
                << std::endl;
      std::cout << " Grad_or_threshold: "
                << grad_or_threshold.reveal<T>(FEATURES_PARTY) << std::endl;
    } else if constexpr (debug) {
      emp_clipping_threshold.reveal<T>(FEATURES_PARTY);
      choose.reveal<bool>(FEATURES_PARTY);
      grad_or_threshold.reveal<T>(FEATURES_PARTY);
    }
  }
}

template <typename T, int Bitwidth>
class ClipAndNoiseFeaturesParty : public OpKernel {
 private:
  int port{0};
  std::string host{""};

 public:
  explicit ClipAndNoiseFeaturesParty(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("StartPort", &port));
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("LabelPartyHost", &host));

    OP_REQUIRES(op_ctx, port > 0,
                InvalidArgument("Port must be a positive integer"));
    OP_REQUIRES(op_ctx, !host.empty(),
                InvalidArgument("Host must be a non-empty string"));
  }

  void Compute(OpKernelContext* op_ctx) override {
    if constexpr (debug) {
      std::cout << "FEATURES_PARTY connecting to " << this->host << ":" << port
                << std::endl;
    }
    NetIO* io = new NetIO(this->host.c_str(), port);
    OP_REQUIRES(op_ctx, io != nullptr,
                InvalidArgument("Failed to create NetIO"));
    setup_semi_honest(io, FEATURES_PARTY);
    if constexpr (debug) {
      std::cout << "FEATURES_PARTY connected" << std::endl;
    }

    // Get the input tensors.
    Tensor const& masks_tensor = op_ctx->input(0);
    auto flat_masks = masks_tensor.flat<T>();
    int num_masks = flat_masks.dimension(0);

    // Setup the output.
    Tensor* output;
    auto output_shape = masks_tensor.shape();
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    // Setup dummy inputs for the MPC protocol.
    std::vector<T> zeros(num_masks, 0);

    // Run the MPC protocol.
    ClipAndNoise<T, Bitwidth, FEATURES_PARTY>(num_masks, flat_masks.data(),
                                              zeros.data(), 0, zeros.data(),
                                              output->flat<T>().data());

    finalize_semi_honest();
    io->flush();
    delete io;
    return;
  }
};

template <typename T, int Bitwidth>
class ClipAndNoiseLabelsParty : public OpKernel {
 private:
  int port{0};
  std::string host{""};

 public:
  explicit ClipAndNoiseLabelsParty(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("StartPort", &port));
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("FeaturePartyHost", &host));

    OP_REQUIRES(op_ctx, port > 0,
                InvalidArgument("Port must be a positive integer"));
    OP_REQUIRES(op_ctx, !host.empty(),
                InvalidArgument("Host must be a non-empty string"));
  }

  void Compute(OpKernelContext* op_ctx) override {
    if constexpr (debug) {
      std::cout << "LABELS_PARTY serving on " << this->host << ":" << port
                << std::endl;
    }
    NetIO* io = new NetIO(nullptr, port);
    OP_REQUIRES(op_ctx, io != nullptr,
                InvalidArgument("Failed to create NetIO"));
    setup_semi_honest(io, LABELS_PARTY);
    if constexpr (debug) {
      std::cout << "LABELS_PARTY connected" << std::endl;
    }

    // Get the input tensors.
    Tensor const& masked_grads_tensor = op_ctx->input(0);
    Tensor const& clipping_threshold_tensor = op_ctx->input(1);
    Tensor const& noises_tensor = op_ctx->input(2);

    // Check the shapes of the input tensors.
    OP_REQUIRES(op_ctx, masked_grads_tensor.shape() == noises_tensor.shape(),
                InvalidArgument(
                    "Masked gradients and noise must have the same shape."));
    OP_REQUIRES(op_ctx,
                TensorShapeUtils::IsScalar(clipping_threshold_tensor.shape()),
                InvalidArgument("Clipping threshold must be a scalar tensor"));

    auto flat_masked_grads = masked_grads_tensor.flat<T>();
    T clipping_threshold = clipping_threshold_tensor.scalar<T>()();
    auto flat_noises = noises_tensor.flat<T>();

    // Setup dummy inputs for the MPC protocol.
    int num_grads = flat_masked_grads.dimension(0);
    std::vector<T> zeros(num_grads, 0);

    // Run the MPC protocol.
    ClipAndNoise<T, Bitwidth, LABELS_PARTY>(
        num_grads, zeros.data(), flat_masked_grads.data(), clipping_threshold,
        flat_noises.data(), nullptr);

    finalize_semi_honest();
    io->flush();
    delete io;
    return;
  }
};

#define REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, Bitwidth)                     \
  REGISTER_KERNEL_BUILDER(Name("ClipAndNoiseFeaturesParty")                   \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TfT>("Dtype")                   \
                              .AttrConstraint<int64_t>("Bitwidth", Bitwidth), \
                          ClipAndNoiseFeaturesParty<T, Bitwidth>);            \
  REGISTER_KERNEL_BUILDER(Name("ClipAndNoiseLabelsParty")                     \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TfT>("Dtype")                   \
                              .AttrConstraint<int64_t>("Bitwidth", Bitwidth), \
                          ClipAndNoiseLabelsParty<T, Bitwidth>);

#define REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart) \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 1);       \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 2);       \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 3);       \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 4);       \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 5);       \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 6);       \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 7);       \
  REGISTER_CLIP_AND_NOISE_KERNELS(T, TfT, BitwidthStart + 8);

REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int64_t, int64, 32);
REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int64_t, int64, 40);
REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int64_t, int64, 48);
REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int64_t, int64, 56);

REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int32_t, int32, 0);
REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int32_t, int32, 8);
REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int32_t, int32, 16);
REGISTER_BYTE_CLIP_AND_NOISE_KERNELS(int32_t, int32, 24);