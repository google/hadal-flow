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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
// #include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "context_variant.h"
#include "discrete_gaussian_sampler.h"
#include "shell_encryption/context.h"
#include "shell_encryption/integral_types.h"
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_bgv_ciphertext.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/status_macros.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using rlwe::SecurePrng;
using rlwe::Uint64;

using tensorflow::DEVICE_CPU;
using tensorflow::int64;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::uint64;
using tensorflow::errors::InvalidArgument;

// Discrete gaussian sampling protocol, based on ``Gaussian Sampling over the
// Integers: Efficient, Generic, Constant-Time'' by Daniele Micciancio and
// Michael Walter, https://eprint.iacr.org/2017/259. The sampling is analogous
// to algorithm SampleCenteredGaussian() in the paper is broken into two steps.
//
// 1) Samples are taken from a base distribution (DG with small scale) and
// combined to generate samples of larger scales in SampleCeneteredGaussianLOP.
// This is analogous to SampleI() in the paper for i=0...n where n is determined
// by the maximum scale of the output distribution.
//
// 2) Two selection vectors are generated, analogous to z and z-1 in
// SampleCenteredGaussian() from the paper.
//
// To compute the final sample of the requested distribution, the caller may
// compute the sum of inner products as follows:
//
// a, b = sample_centered_gaussian_f(...)
// samples_a = sample_centered_gaussian_l(...)
// samples_b = sample_centered_gaussian_l(...)
// final_sample = a * samples_a + b * samples_b
//
// where * represents matrix multiplication.

template <typename SamplerT>
class SampleCenteredGaussianFOp : public OpKernel {
 private:
  float scale;
  float base_scale;
  float max_scale;

 public:
  explicit SampleCenteredGaussianFOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("base_scale", &base_scale));
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("max_scale", &max_scale));

    OP_REQUIRES(op_ctx, base_scale < max_scale,
                InvalidArgument("Base scale must be less than max scale."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    OP_REQUIRES_VALUE(float scale, op_ctx, GetScalar<float>(op_ctx, 0));
    int n_i = 0, n_max = 0;
    double s_i = 0, s_max = 0;

    // n_i is the number of iterations SampleI() must run to compute a sample
    // with the requested scale, and s_i is an internal parameter of the
    // algorithm from the paper.
    OP_REQUIRES_VALUE(
        std::tie(n_i, s_i), op_ctx,
        DiscreteGaussianSampler<SamplerT>::NumIterations(scale, base_scale));

    // n_max and s_max are as above, but for the maximum supported scale.
    OP_REQUIRES_VALUE(std::tie(n_max, s_max), op_ctx,
                      DiscreteGaussianSampler<SamplerT>::NumIterations(
                          max_scale, base_scale));
    // The number of samples which must be scaled is one larger than the number
    // of iterations
    n_max += 1;

    // Allocate the output tensor.
    TensorShape output_shape;
    OP_REQUIRES_OK(op_ctx,
                   TensorShape::BuildTensorShape({n_max}, &output_shape));
    Tensor *a, *b;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &a));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(1, output_shape, &b));
    auto flat_a = a->flat<int64_t>();
    auto flat_b = b->flat<int64_t>();

    double t = static_cast<double>(scale) / static_cast<double>(s_i);
    double z_hat = std::ceil(.5 * (1 + std::sqrt(2. * t * t - 1)));

    for (int i = 0; i < n_max; ++i) {
      if (i == n_i) {
        flat_a(i) = z_hat;
        flat_b(i) = z_hat - 1;
      } else {
        flat_a(i) = 0;
        flat_b(i) = 0;
      }
    }
  }
};

template <typename T, typename SamplerT>
class SampleCenteredGaussianLOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Prng = rlwe::SecurePrng;

  int64 num_samples;
  float base_scale;
  float max_scale;
  std::unique_ptr<DiscreteGaussianSampler<SamplerT>> sampler;

 public:
  explicit SampleCenteredGaussianLOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("base_scale", &base_scale));
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("max_scale", &max_scale));

    OP_REQUIRES_VALUE(sampler, op_ctx,
                      DiscreteGaussianSampler<SamplerT>::Create(base_scale));
  }

  void Compute(OpKernelContext* op_ctx) override {
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    // Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    Prng* prng = shell_ctx_var->prng_[0].get();

    OP_REQUIRES_VALUE(int64 num_samples, op_ctx, GetScalar<int64>(op_ctx, 1));
    OP_REQUIRES(op_ctx, num_samples > 0,
                InvalidArgument("Number of samples must be positive."));

    int n_max = 0, s_max = 0;
    OP_REQUIRES_VALUE(std::tie(n_max, s_max), op_ctx,
                      sampler->NumIterations(max_scale, sampler->s_base_));
    n_max += 1;

    // Allocate the output tensor.
    TensorShape output_shape;
    OP_REQUIRES_OK(op_ctx, TensorShape::BuildTensorShape({num_samples, n_max},
                                                         &output_shape));
    Tensor* samples;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &samples));
    auto flat_samples = samples->shaped<int64_t, 2>({num_samples, n_max});

    // Run SampleI() for each sample. Instead of SampleI() returning only the
    // largest sample, as done in the paper, returns the intermediate sample for
    // each i, stored by increasing scale for each sample.
    for (int64_t i = 0; i < num_samples; ++i) {
      OP_REQUIRES_VALUE(auto sample_tree, op_ctx,
                        sampler->SampleIIterative(*prng, n_max));
      for (int j = 0; j < n_max; ++j) {
        OP_REQUIRES(
            op_ctx, j < 64,
            InvalidArgument("Internal error: invalid shift amount: ", j));
        uint64_t tree_index = sample_tree.size() - (1ULL << j);
        OP_REQUIRES(
            op_ctx, tree_index >= 0 && tree_index < sample_tree.size(),
            InvalidArgument("Internal error: invalid tree index: ", tree_index,
                            " for sample ", i, " and j ", j,
                            ". sample_tree size: ", sample_tree.size()));
        flat_samples(i, j) = sample_tree[tree_index];
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SampleCenteredGaussianF64").Device(DEVICE_CPU),
                        SampleCenteredGaussianFOp<Uint64>);

REGISTER_KERNEL_BUILDER(Name("SampleCenteredGaussianL64").Device(DEVICE_CPU),
                        SampleCenteredGaussianLOp<uint64, Uint64>);
