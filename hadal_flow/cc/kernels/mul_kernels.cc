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
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_polynomial.h"
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

template <typename T>
class MulCtCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit MulCtCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    BCast bcast(BCast::FromShape(a.shape()), BCast::FromShape(b.shape()),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(
        op_ctx, bcast.IsValid(),
        InvalidArgument("Invalid broadcast between ", a.shape().DebugString(),
                        " and ", b.shape().DebugString()));
    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat<Variant>();
    IndexConverterFunctor a_bcaster(bcast.output_shape(), a.shape());
    IndexConverterFunctor b_bcaster(bcast.output_shape(), b.shape());

    // Recover num_slots from first ciphertext.
    SymmetricCtVariant<T> const* ct_var =
        flat_a(0).get<SymmetricCtVariant<T>>();
    OP_REQUIRES(
        op_ctx, ct_var != nullptr,
        InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
    OP_REQUIRES_OK(
        op_ctx, const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
    SymmetricCt const& ct = ct_var->ct;
    int num_slots = 1 << ct.LogN();
    int num_components = ct.NumModuli();

    // Allocate the output tensor which is the same shape as each of the inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    // Multiply each pair of ciphertexts and store the result in the output.
    auto mul_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<T> const* ct_a_var =
            flat_a(a_bcaster(i)).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(
            op_ctx, ct_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index:", i,
                            " for input a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct_a = ct_a_var->ct;

        SymmetricCtVariant<T> const* ct_b_var =
            flat_b(b_bcaster(i)).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(
            op_ctx, ct_b_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index:", i,
                            " for input b did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_b_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct_b = ct_b_var->ct;

        OP_REQUIRES_VALUE(SymmetricCt ct_c, op_ctx, ct_a * ct_b);

        // Wrap the result in a SymmetricCtVariant and store it in the output.
        // SHELL's multiplication preserves moduli pointers of the first input.
        // Ensure the output holds smart pointers to the input's context to
        // prevent premature deletion of the moduli.
        SymmetricCtVariant ct_c_var(std::move(ct_c), ct_a_var->ct_context,
                                    ct_a_var->error_params);
        flat_output(i) = std::move(ct_c_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_mul = 30 * num_slots * num_components;
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_mul,
                             mul_in_range);
  }
};

template <typename T>
class MulCtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit MulCtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    BCast bcast(BCast::FromShape(a.shape()), BCast::FromShape(b.shape()),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(
        op_ctx, bcast.IsValid(),
        InvalidArgument("Invalid broadcast between ", a.shape().DebugString(),
                        " and ", b.shape().DebugString()));
    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat<Variant>();
    IndexConverterFunctor a_bcaster(bcast.output_shape(), a.shape());
    IndexConverterFunctor b_bcaster(bcast.output_shape(), b.shape());

    // Allocate the output tensor which is the same shape as each of the inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    // Recover num_slots from first ciphertext.
    SymmetricCtVariant<T> const* ct_var =
        flat_a(0).get<SymmetricCtVariant<T>>();
    OP_REQUIRES(
        op_ctx, ct_var != nullptr,
        InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
    OP_REQUIRES_OK(
        op_ctx, const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
    SymmetricCt const& ct = ct_var->ct;
    int num_slots = 1 << ct.LogN();
    int num_components = ct.NumModuli();

    auto mul_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<T> const* ct_a_var =
            flat_a(a_bcaster(i)).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(
            op_ctx, ct_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index:", i,
                            " for input a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct_a = ct_a_var->ct;

        PolynomialVariant<T> const* pv_b_var =
            flat_b(b_bcaster(i)).get<PolynomialVariant<T>>();
        OP_REQUIRES(
            op_ctx, pv_b_var != nullptr,
            InvalidArgument("PolynomialVariant at flat index:", i,
                            " for input b did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<PolynomialVariant<T>*>(pv_b_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_));
        RnsPolynomial const& pt_b = pv_b_var->poly;

        OP_REQUIRES_VALUE(SymmetricCt ct_c, op_ctx,
                          ct_a * pt_b);  // shell absorb operation

        // Wrap the result in a SymmetricCtVariant and store it in the output.
        // The output ct will hold raw pointers to moduli stored in the  input's
        // context. Ensure the output ciphertext Variant wrapper holds smart
        // pointers to the input's context to prevent premature deletion of the
        // moduli
        SymmetricCtVariant ct_c_var(std::move(ct_c), ct_a_var->ct_context,
                                    ct_a_var->error_params);
        flat_output(i) = std::move(ct_c_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_mul = 30 * num_slots * num_components;
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_mul,
                             mul_in_range);
  }
};

// This Op can multiply either a shell ciphertext or a plaintext polynomial by
// a plaintext scalar, depending on the class template.
template <typename T, typename PtT, typename CtOrPolyVariant>
class MulShellTfScalarOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;

  float scaling_factor_ = 1.;
  bool random_round_ = false;

 public:
  explicit MulShellTfScalarOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("scaling_factor", &scaling_factor_));
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("random_round", &random_round_));

    OP_REQUIRES(op_ctx, scaling_factor_ > 0,
                InvalidArgument("scaling_factor must be positive."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    Encoder const* encoder = shell_ctx_var->encoder_.get();

    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    BCast bcast(BCast::FromShape(a.shape()), BCast::FromShape(b.shape()),
                /*fewer_dims_optimization=*/false);
    OP_REQUIRES(
        op_ctx, bcast.IsValid(),
        InvalidArgument("Invalid broadcast between ", a.shape().DebugString(),
                        " and ", b.shape().DebugString()));
    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat<PtT>();
    IndexConverterFunctor a_bcaster(bcast.output_shape(), a.shape());
    IndexConverterFunctor b_bcaster(bcast.output_shape(), b.shape());

    // Recover num_slots from first or plaintext.
    int num_slots;
    int num_components;
    if constexpr (std::is_same<CtOrPolyVariant, SymmetricCtVariant<T>>::value) {
      SymmetricCtVariant<T> const* ct_var =
          flat_a(0).get<SymmetricCtVariant<T>>();
      OP_REQUIRES(
          op_ctx, ct_var != nullptr,
          InvalidArgument("SymmetricCtVariant a did not unwrap successfully."));
      OP_REQUIRES_OK(
          op_ctx,
          const_cast<SymmetricCtVariant<T>*>(ct_var)->MaybeLazyDecode(
              shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
      SymmetricCt const& ct = ct_var->ct;
      num_slots = 1 << ct.LogN();
      num_components = ct.NumModuli();
    } else {
      PolynomialVariant<T> const* pt_var =
          flat_a(0).get<PolynomialVariant<T>>();
      OP_REQUIRES(
          op_ctx, pt_var != nullptr,
          InvalidArgument("PolynomialVariant a did not unwrap successfully."));
      OP_REQUIRES_OK(op_ctx,
                     const_cast<PolynomialVariant<T>*>(pt_var)->MaybeLazyDecode(
                         shell_ctx_var->ct_context_));
      RnsPolynomial const& pt = pt_var->poly;
      num_slots = 1 << pt.LogN();
      num_components = 1;
    }

    // Allocate the output tensor which is the same shape as the first input.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    // Now multiply.
    auto mul_in_range = [&](int start, int end, int worker_id) {
      for (int i = start; i < end; ++i) {
        // First multiply the scalar b by the scaling factor.
        double val =
            static_cast<double>(flat_b(b_bcaster(i))) * scaling_factor_;

        if (random_round_) {
          // Use a separate PRNG for each thread to avoid contention.
          int prng_i = worker_id % shell_ctx_var->prng_.size();
          SecurePrng* prng = shell_ctx_var->prng_[prng_i].get();
          OP_REQUIRES_VALUE(val, op_ctx, RandomRound<T>(prng, val));
        } else {
          // Just round to the nearest integer.
          val = round(val);
        }

        // Encode the scalar b, taking the scaling factor into account.
        T wrapped_b{};
        EncodeScalar(op_ctx, static_cast<PtT>(round(val)), encoder, &wrapped_b);

        CtOrPolyVariant const* ct_or_pt_var =
            flat_a(a_bcaster(i)).get<CtOrPolyVariant>();
        OP_REQUIRES(
            op_ctx, ct_or_pt_var != nullptr,
            InvalidArgument("Input at flat index:", i,
                            " for input a did not unwrap successfully."));

        if constexpr (std::is_same<CtOrPolyVariant,
                                   PolynomialVariant<T>>::value) {
          OP_REQUIRES_OK(op_ctx,
                         const_cast<PolynomialVariant<T>*>(ct_or_pt_var)
                             ->MaybeLazyDecode(shell_ctx_var->ct_context_));
          RnsPolynomial const& poly = ct_or_pt_var->poly;

          OP_REQUIRES_VALUE(RnsPolynomial result, op_ctx,
                            poly.Mul(wrapped_b, shell_ctx->MainPrimeModuli()));

          PolynomialVariant<T> result_var(std::move(result),
                                          shell_ctx_var->ct_context_);
          flat_output(i) = std::move(result_var);
        } else if constexpr (std::is_same<CtOrPolyVariant,
                                          SymmetricCtVariant<T>>::value) {
          OP_REQUIRES_OK(op_ctx,
                         const_cast<SymmetricCtVariant<T>*>(ct_or_pt_var)
                             ->MaybeLazyDecode(shell_ctx_var->ct_context_,
                                               shell_ctx_var->error_params_));
          SymmetricCt const& ct = ct_or_pt_var->ct;

          OP_REQUIRES_VALUE(SymmetricCt result, op_ctx,
                            ct * wrapped_b);  // shell aborb operation

          // The output ct will hold raw pointers to moduli stored in the
          // input's context. Ensure the output ciphertext Variant wrapper holds
          // smart pointers to the input's context to prevent premature deletion
          // of the moduli
          SymmetricCtVariant result_var(std::move(result),
                                        ct_or_pt_var->ct_context,
                                        ct_or_pt_var->error_params);
          flat_output(i) = std::move(result_var);
        }
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_mul = 20 * num_slots * num_components;
    thread_pool->ParallelForWithWorkerId(
        flat_output.dimension(0),
        tsl::thread::ThreadPool::SchedulingParams::Adaptive(cost_per_mul),
        mul_in_range);
  }

 private:
  void EncodeScalar(OpKernelContext* op_ctx, PtT const& val,
                    Encoder const* encoder, T* wrapped_val) {
    if constexpr (std::is_signed<PtT>::value) {
      // SHELL is built on the assumption that the plaintext type (in this
      // case `PtT`) will always fit into the ciphertext underlying type
      // (in this case `T`). E.g. the plaintext modulus is stored as the
      // ciphertext type. This is true even in the RNS code paths. This means
      // that this function can convert `PtT` to a signed version of `T`,
      // then modulus switch into plaintext field t and type `T` without
      // overflow.
      using SignedInteger = std::make_signed_t<T>;

      SignedInteger signed_val = static_cast<SignedInteger>(val);

      // Map signed integers into the plaintext modulus field.
      OP_REQUIRES_VALUE(
          std::vector<T> wrapped_val_vector, op_ctx,
          (encoder->template WrapSigned<SignedInteger>({signed_val})));

      *wrapped_val = wrapped_val_vector[0];
    } else {
      // Since From and To are both unsigned, just cast and copy.
      *wrapped_val = static_cast<T>(val);
    }
  }
};

template <typename T>
class MulPtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit MulPtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    BCast bcast(BCast::FromShape(a.shape()), BCast::FromShape(b.shape()),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(
        op_ctx, bcast.IsValid(),
        InvalidArgument("Invalid broadcast between ", a.shape().DebugString(),
                        " and ", b.shape().DebugString()));
    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat<Variant>();
    IndexConverterFunctor a_bcaster(bcast.output_shape(), a.shape());
    IndexConverterFunctor b_bcaster(bcast.output_shape(), b.shape());

    // Recover num_slots from first plaintext.
    PolynomialVariant<T> const* pt_var = flat_a(0).get<PolynomialVariant<T>>();
    OP_REQUIRES(
        op_ctx, pt_var != nullptr,
        InvalidArgument("PolynomialVariant a did not unwrap successfully."));
    OP_REQUIRES_OK(op_ctx,
                   const_cast<PolynomialVariant<T>*>(pt_var)->MaybeLazyDecode(
                       shell_ctx_var->ct_context_));
    RnsPolynomial const& pt = pt_var->poly;
    int num_slots = 1 << pt.LogN();

    // Allocate the output tensor which is the same shape as each of the
    // inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    auto add_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        PolynomialVariant<T> const* pv_a_var =
            flat_a(a_bcaster(i)).get<PolynomialVariant<T>>();
        OP_REQUIRES(
            op_ctx, pv_a_var != nullptr,
            InvalidArgument("PolynomialVariant at flat index:", i,
                            " for input a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<PolynomialVariant<T>*>(pv_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_));
        RnsPolynomial const& pt_a = pv_a_var->poly;

        PolynomialVariant<T> const* pv_b_var =
            flat_b(b_bcaster(i)).get<PolynomialVariant<T>>();
        OP_REQUIRES(
            op_ctx, pv_b_var != nullptr,
            InvalidArgument("PolynomialVariant at flat index:", i,
                            " for input b did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<PolynomialVariant<T>*>(pv_b_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_));
        RnsPolynomial const& pt_b = pv_b_var->poly;

        OP_REQUIRES_VALUE(RnsPolynomial pt_c, op_ctx,
                          pt_a.Mul(pt_b, shell_ctx->MainPrimeModuli()));

        PolynomialVariant<T> pt_c_var(std::move(pt_c),
                                      shell_ctx_var->ct_context_);
        flat_output(i) = std::move(pt_c_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_add = 20 * num_slots;
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_add,
                             add_in_range);
  }
};

template <typename T, typename PtT>
class MatMulCtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;

 public:
  explicit MatMulCtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  // A matrix of ciphertext multiplied by plaintext matrix can be performed
  // without any expensive ciphertext slot rotations and without needing to
  // encode the plaintext into a polynomial. This only uses ciphertext *
  // plaintext scalar multiplication and addition (i.e. no ciphertext rotations)
  // by doing the following:
  //
  // def simple_matmul_ct_pt(x, y):
  //  xm, xn = x.shape
  //  ym, yn = y.shape
  //
  //  result = tf.zeros([xm, yn])
  //
  //  for i in range(len(yn)):
  //    result_column_i = tf.zeros([xm, 1])
  //    for j in range(len(ym)):
  //      result_column_i += x_polynomial_column[j] * y_scalar[j, i]
  //
  // TODO(jchoncholas): Support batching. a's dimension may be greater
  // than 2 and matmul is performed for each of the outer dims.
  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));

    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    // TODO if debug
    OP_REQUIRES(op_ctx, shell_ctx != nullptr,
                InvalidArgument("Shell context object is empty."));

    Encoder const* encoder = shell_ctx_var->encoder_.get();
    // TODO if debug
    OP_REQUIRES(op_ctx, encoder != nullptr,
                InvalidArgument("Shell encoder object is empty."));

    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    // a is a vector of Polynomials so first dimension is the number of slots.
    OP_REQUIRES(op_ctx, a.dims() == 1 && b.dims() == 2,
                InvalidArgument("Inputs must have dimension 2."));

    OP_REQUIRES(op_ctx, a.dim_size(a.dims() - 1) == b.dim_size(0),
                InvalidArgument(
                    "Inputs dimensions do not support matrix multiplication."));

    // Output is a vector of Polynomials and the first dimension is the number
    // of slots.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(
                               0, TensorShape{b.dim_size(1)}, &output));

    // Set up the flat views of the inputs and output tensors.
    auto flat_a = a.flat<Variant>();
    auto flat_b = b.flat_outer_dims<PtT>();
    auto flat_output = output->flat<Variant>();

    auto ct_col_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<T> const* ct_a_var =
            flat_a(0).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(
            op_ctx, ct_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index: 0",
                            " for input a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct_a = ct_a_var->ct;

        // Before multiplying, the check if the plaintext integer is signed.
        // If so, it needs to be imported into the field of the plaintext
        // modulus to properly handle negative values.
        OP_REQUIRES_VALUE(T mint_b, op_ctx,
                          ToSigned(flat_b(0, i), encoder, op_ctx));

        OP_REQUIRES_VALUE(SymmetricCt ct_result, op_ctx,
                          ct_a * mint_b);  // ciphertext * scalar

        for (int j = 1; j < b.dim_size(0); ++j) {
          SymmetricCtVariant<T> const* ct_a_var =
              flat_a(j).get<SymmetricCtVariant<T>>();
          OP_REQUIRES(
              op_ctx, ct_a_var != nullptr,
              InvalidArgument("SymmetricCtVariant at flat index:", j,
                              " for input a did not unwrap successfully."));
          OP_REQUIRES_OK(
              op_ctx,
              const_cast<SymmetricCtVariant<T>*>(ct_a_var)->MaybeLazyDecode(
                  shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
          SymmetricCt const& ct_a = ct_a_var->ct;

          // Again check if the plaintext integer is signed.
          OP_REQUIRES_VALUE(T mint_b, op_ctx,
                            ToSigned(flat_b(j, i), encoder, op_ctx));

          OP_REQUIRES_VALUE(SymmetricCt scaled, op_ctx,
                            ct_a * mint_b);  // Ct * scalar
          OP_REQUIRES_OK(op_ctx, ct_result.AddInPlace(scaled));
        }

        SymmetricCtVariant ct_result_var(
            std::move(ct_result), ct_a_var->ct_context, ct_a_var->error_params);
        flat_output(i) = std::move(ct_result_var);
      }
    };
    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_outter =
        32384 * b.dim_size(0);  // ns, measured on log_n = 11
    thread_pool->ParallelFor(b.dim_size(1), cost_per_outter, ct_col_in_range);
  }

  static StatusOr<T> ToSigned(PtT const& val, Encoder const* encoder,
                              OpKernelContext* op_ctx) {
    if constexpr (std::is_signed<PtT>::value) {
      // SHELL is built on the assumption that the plaintext type (in this
      // case `From`) will always fit into the ciphertext underlying type
      // (in this case `To`). E.g. the plaintext modulus is stored as the
      // ciphertext type. This is true even in the RNS code paths. This means
      // that this function can convert `From` to a signed version of `To`,
      // then modulus switch into plaintext field t and type `To` without
      // overflow.
      using SignedInteger = std::make_signed_t<T>;

      // Map signed integer into the unsigned plaintext modulus field.
      auto signed_val = (encoder->template WrapSigned<SignedInteger>({val}));

      if (!signed_val.ok()) {
        return signed_val.status();
      } else {
        return signed_val.value()[0];
      }
    } else {
      // Since T and PtT are both unsigned, just cast and copy.
      return static_cast<T>(val);
    }
  }
};

template <typename PtT, typename T>
class MatMulPtCtOp : public OpKernel {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using Modulus = rlwe::PrimeModulus<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;

  std::string reduction;
  char const* galois_reduction = "galois";
  char const* fast_reduction = "fast";
  char const* no_reduction = "none";

 public:
  explicit MatMulPtCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("reduction", &reduction));
    OP_REQUIRES(
        op_ctx,
        reduction == "galois" || reduction == "fast" || reduction == "none",
        InvalidArgument("Invalid reduction attribute: ", reduction,
                        ". Must be 'galois', 'fast', or 'none'."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Get the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));

    Context const* shell_ctx = shell_ctx_var->ct_context_.get();
    auto const& sub_powers = shell_ctx_var->substitution_powers_;
    OP_REQUIRES(op_ctx, shell_ctx != nullptr,
                InvalidArgument("Shell context object is empty."));
    Encoder const* encoder = shell_ctx_var->encoder_.get();
    // TODO if debug
    OP_REQUIRES(op_ctx, encoder != nullptr,
                InvalidArgument("Shell encoder object is empty."));

    Tensor const& a = op_ctx->input(1);
    Tensor const& b = op_ctx->input(2);

    // Rotation keys are only required if fast rotations are enabled.
    RotationKeyVariant<T> const* rotation_key_var = nullptr;
    if (reduction == galois_reduction) {
      OP_REQUIRES_VALUE(rotation_key_var, op_ctx,
                        GetVariant<RotationKeyVariant<T>>(op_ctx, 3));
      OP_REQUIRES(
          op_ctx, rotation_key_var != nullptr,
          InvalidArgument("RotationKeyVariant did not unwrap successfully."));
      OP_REQUIRES_OK(op_ctx,
                     const_cast<RotationKeyVariant<T>*>(rotation_key_var)
                         ->MaybeLazyDecode(shell_ctx_var->ct_context_));
    } else {
      (void)rotation_key_var;
    }
    std::vector<std::shared_ptr<RotationKey>> empty_rot_keys{};
    std::vector<std::shared_ptr<RotationKey>> const& rot_keys =
        reduction == galois_reduction ? rotation_key_var->keys : empty_rot_keys;

    // b is a vector of Polynomials so first dimension is the number of
    // slots.
    OP_REQUIRES(op_ctx, a.dims() >= 2 && b.dims() == 1,
                InvalidArgument("Inputs must have dimension 2."));

    // Extract the dimension sizes from the input tensors.
    int num_slots = 1 << shell_ctx->LogN();
    int num_ct_cols = b.dim_size(0);
    int pt_total_size = a.NumElements();
    int num_pt_inner_rows = a.dim_size(a.dims() - 2);
    int num_pt_inner_cols = a.dim_size(a.dims() - 1);
    int num_pt_outer_dims =
        pt_total_size / num_pt_inner_rows / num_pt_inner_cols;

    OP_REQUIRES(op_ctx, num_pt_inner_cols == num_slots,
                InvalidArgument(
                    "Inputs dimensions do not support matrix multiplication."));

    // Output of one plaintext matrix * ciphertext matrix is a matrix of
    // *polynomials*. This is different from how plaintext matrix multiplication
    // works where the output is just a matrix of integers. In this case, we
    // have effectively increased the number of dimensions of the output by 1.
    // I.e. the inner dimension is a ciphertext where *every* slot contains
    // the same value. This is due to the reduce_sum operation which
    // results in ciphertexts whose slots are all the same value, the result
    // of the reduce sum.
    Tensor* output;
    auto output_shape = a.shape();
    OP_REQUIRES_OK(op_ctx, output_shape.RemoveLastDimsWithStatus(1));
    OP_REQUIRES_OK(op_ctx, output_shape.AddDimWithStatus(num_ct_cols));
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto flat_a = a.shaped<PtT, 3>(
        {num_pt_outer_dims, num_pt_inner_rows, num_pt_inner_cols});

    // Printing the shapes of the tensors changes whether it is a Tensorflow
    // Tensor or an Eigen Tensor, which is what is retured from the flat()
    // calls.
    // std::cout << "a shape = " << a.shape().DebugString() << std::endl;
    // std::cout << "flat_a shape = " << flat_a.dimensions() << std::endl;

    auto flat_b = b.flat<Variant>();
    auto flat_output = output->shaped<Variant, 3>(
        {num_pt_outer_dims, num_pt_inner_rows, num_ct_cols});

    // Extract the first ciphertext in b to recover the moduli. This is used
    // to create new ciphertexts after fast rotations and note the moduli must
    // come from the ciphertext, not the shell context, to ensure smart pointers
    // are properly preserved.
    SymmetricCtVariant<T> const* first_ct_b_var =
        flat_b(0).get<SymmetricCtVariant<T>>();
    OP_REQUIRES(op_ctx, first_ct_b_var != nullptr,
                InvalidArgument("SymmetricCtVariant at flat index: 0",
                                " for input b did not unwrap successfully."));
    OP_REQUIRES_OK(op_ctx, const_cast<SymmetricCtVariant<T>*>(first_ct_b_var)
                               ->MaybeLazyDecode(shell_ctx_var->ct_context_,
                                                 shell_ctx_var->error_params_));
    auto const& main_moduli = first_ct_b_var->ct.Moduli();
    std::vector<Modulus const*> main_moduli_vector;
    main_moduli_vector.assign(main_moduli.begin(), main_moduli.end());

    // Setup constants used in parallelizing the computation.
    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_inner =
        30000 * num_ct_cols / num_slots / 2;  // ns, measured on log_n = 11

    // For each outer n-2 dimensions of a, perform the matrix multiplication
    // on the inner dimension.
    for (int outer = 0; outer < num_pt_outer_dims; ++outer) {
      // ParallelFor in the inner dimensions of the plaintext matrix.
      auto pt_inner_dim_in_range = [&](int start, int end) {
        // Iterate over the rows in the plaintext inner matrix, encode the
        // plaintext matrix into polynomials row-wise then multiply by the
        // ciphertext column.
        for (int i = start; i < end; ++i) {
          // Encode the row of the plaintext matrix into a polynomial.
          std::vector<T> wrapped_row;
          if constexpr (std::is_signed<PtT>::value) {
            // SHELL is built on the assumption that the plaintext type (in
            // this case `From`) will always fit into the ciphertext
            // underlying type (in this case `To`). E.g. the plaintext modulus
            // is stored as the ciphertext type. This is true even in the RNS
            // code paths. This means that this function can convert `From` to
            // a signed version of `To`, then modulus switch into plaintext
            // field t and type `To` without overflow.
            using SignedInteger = std::make_signed_t<T>;

            // Copy into contiguous memory of signed `To`'s
            std::vector<SignedInteger> nums(num_slots);
            for (int slot = 0; slot < num_slots; ++slot) {
              nums[slot] = static_cast<SignedInteger>(flat_a(outer, i, slot));
            }

            // Map signed integers into the plaintext modulus field.
            OP_REQUIRES_VALUE(
                wrapped_row, op_ctx,
                (encoder->template WrapSigned<SignedInteger>(nums)));
          } else {
            wrapped_row = std::vector<T>(num_slots);
            // Since From and To are both unsigned, just cast and copy.
            for (int slot = 0; slot < num_slots; ++slot) {
              wrapped_row[slot] = static_cast<T>(flat_a(outer, i, slot));
            }
          }

          OP_REQUIRES_VALUE(RnsPolynomial row_polynomial, op_ctx,
                            encoder->EncodeBgv(wrapped_row, main_moduli));

          // Multiply the row by each of the ciphertext vector,
          // point - wise.
          for (int ct_col = 0; ct_col < num_ct_cols; ++ct_col) {
            // Index into the ciphertext columns.
            SymmetricCtVariant<T> const* ct_b_var =
                flat_b(ct_col).get<SymmetricCtVariant<T>>();
            // TODO if debug
            OP_REQUIRES(
                op_ctx, ct_b_var != nullptr,
                InvalidArgument("SymmetricCtVariant at flat index: 0",
                                " for input a did not unwrap successfully."));
            OP_REQUIRES_OK(
                op_ctx,
                const_cast<SymmetricCtVariant<T>*>(ct_b_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
            SymmetricCt const& ct_b = ct_b_var->ct;

            // Perform the multiplication.
            OP_REQUIRES_VALUE(SymmetricCt ct_result, op_ctx,
                              ct_b * row_polynomial);

            // Reduce sum the result.
            // Note the ciphertext rotations operate on each half of the
            // ciphertext separately. So the max rotatation is by half the
            // number of slots.
            if (reduction == galois_reduction) {
              for (int shift = 1; shift < num_slots / 2; shift <<= 1) {
                OP_REQUIRES(
                    op_ctx,
                    shift - 1 <
                        static_cast<int>(rot_keys.size()),  // Skip key 0.
                    InvalidArgument("No key for shift of '", shift, "'"));
                RotationKey const* k = rot_keys[shift].get();

                // Rotate by the shift.
                OP_REQUIRES_VALUE(auto ct_sub, op_ctx,
                                  ct_result.Substitute(k->SubstitutionPower()));
                OP_REQUIRES_VALUE(auto ct_rot, op_ctx, k->ApplyTo(ct_sub));

                // Add to the sum.
                OP_REQUIRES_OK(op_ctx, ct_result.AddInPlace(ct_rot));
              }

            } else if (reduction == fast_reduction) {
              OP_REQUIRES_VALUE(
                  RnsPolynomial sum_component_zero, op_ctx,
                  ct_result.Component(0));  // deep copy to start the sum.
              for (int shift = 1; shift < num_slots / 2; shift <<= 1) {
                // Rotate by the shift.
                OP_REQUIRES_VALUE(RnsPolynomial sum_shifted, op_ctx,
                                  sum_component_zero.Substitute(
                                      sub_powers[shift], main_moduli));

                // Add to the sum.
                OP_REQUIRES_OK(op_ctx, sum_component_zero.AddInPlace(
                                           sum_shifted, main_moduli));
              }

              // The second component of the ciphertext is unchanged.
              OP_REQUIRES_VALUE(RnsPolynomial passthrough_component_one, op_ctx,
                                ct_result.Component(1));

              std::vector<RnsPolynomial> components{
                  std::move(sum_component_zero),
                  std::move(passthrough_component_one),
              };

              // Recreate the ciphertext with the new components.
              // TODO(james-choncholas): Noise estimation is not correct.
              ct_result = SymmetricCt{std::move(components), main_moduli_vector,
                                      ct_result.PowerOfS(),
                                      ct_result.Error() * ct_result.LogN(),
                                      ct_result.ErrorParams()};
            }
            // else if no reduction, do nothing.

            // At this point we have one ciphertext per row of the plaintext
            // matrix where every element in the ciphertext is the same value,
            // the result of the reduce sum operation. Store in the output
            // tensor.
            SymmetricCtVariant<T> ct_result_var(std::move(ct_result),
                                                ct_b_var->ct_context,
                                                ct_b_var->error_params);
            flat_output(outer, i, ct_col) = std::move(ct_result_var);
          }
        }
      };

      // Use a parallel for loop in the inner dimensions of the plaintext
      // matrix.
      thread_pool->ParallelFor(num_pt_inner_rows, cost_per_inner * 1000 * 1000,
                               pt_inner_dim_in_range);
    }
  }
};

// Multiply ciphertext by ciphertext.
REGISTER_KERNEL_BUILDER(Name("MulCtCt64").Device(DEVICE_CPU),
                        MulCtCtOp<uint64>);

// Multiply ciphertext by plaintext.
REGISTER_KERNEL_BUILDER(Name("MulCtPt64").Device(DEVICE_CPU),
                        MulCtPtOp<uint64>);

// Multiply plaintext or ciphertext by plaintext scalar.
REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint8>("Dtype"),
    MulShellTfScalarOp<uint64, uint8, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int8>("Dtype"),
    MulShellTfScalarOp<uint64, int8, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint8>("Dtype"),
    MulShellTfScalarOp<uint64, uint8, SymmetricCtVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int8>("Dtype"),
    MulShellTfScalarOp<uint64, int8, SymmetricCtVariant<uint64>>);

REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint16>("Dtype"),
    MulShellTfScalarOp<uint64, uint16, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int16>("Dtype"),
    MulShellTfScalarOp<uint64, int16, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint16>("Dtype"),
    MulShellTfScalarOp<uint64, uint16, SymmetricCtVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int16>("Dtype"),
    MulShellTfScalarOp<uint64, int16, SymmetricCtVariant<uint64>>);

REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint32>("Dtype"),
    MulShellTfScalarOp<uint64, uint32, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int32>("Dtype"),
    MulShellTfScalarOp<uint64, int32, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint32>("Dtype"),
    MulShellTfScalarOp<uint64, uint32, SymmetricCtVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int32>("Dtype"),
    MulShellTfScalarOp<uint64, int32, SymmetricCtVariant<uint64>>);

REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint64>("Dtype"),
    MulShellTfScalarOp<uint64, uint64, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int64>("Dtype"),
    MulShellTfScalarOp<uint64, int64, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<uint64>("Dtype"),
    MulShellTfScalarOp<uint64, uint64, SymmetricCtVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<int64>("Dtype"),
    MulShellTfScalarOp<uint64, int64, SymmetricCtVariant<uint64>>);

REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<float>("Dtype"),
    MulShellTfScalarOp<uint64, float, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulPtTfScalar64").Device(DEVICE_CPU).TypeConstraint<double>("Dtype"),
    MulShellTfScalarOp<uint64, double, PolynomialVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<float>("Dtype"),
    MulShellTfScalarOp<uint64, float, SymmetricCtVariant<uint64>>);
REGISTER_KERNEL_BUILDER(
    Name("MulCtTfScalar64").Device(DEVICE_CPU).TypeConstraint<double>("Dtype"),
    MulShellTfScalarOp<uint64, double, SymmetricCtVariant<uint64>>);

// Multiply plaintext by plaintext.
REGISTER_KERNEL_BUILDER(Name("MulPtPt64").Device(DEVICE_CPU),
                        MulPtPtOp<uint64>);

// Matrix multiply ciphertext and plaintext.
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<uint8>("Dtype"),
    MatMulCtPtOp<uint64, uint8>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int8>("Dtype"),
    MatMulCtPtOp<uint64, int8>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<uint16>("Dtype"),
    MatMulCtPtOp<uint64, uint16>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int16>("Dtype"),
    MatMulCtPtOp<uint64, int16>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<uint32>("Dtype"),
    MatMulCtPtOp<uint64, uint32>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int32>("Dtype"),
    MatMulCtPtOp<uint64, int32>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<uint64>("Dtype"),
    MatMulCtPtOp<uint64, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulCtPt64").Device(DEVICE_CPU).TypeConstraint<int64>("Dtype"),
    MatMulCtPtOp<uint64, int64>);

// Matrix multiply plaintext and ciphertext.
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<uint8>("Dtype"),
    MatMulPtCtOp<uint8, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int8>("Dtype"),
    MatMulPtCtOp<int8, uint64>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<uint16>("Dtype"),
    MatMulPtCtOp<uint16, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int16>("Dtype"),
    MatMulPtCtOp<int16, uint64>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<uint32>("Dtype"),
    MatMulPtCtOp<uint32, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int32>("Dtype"),
    MatMulPtCtOp<int32, uint64>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<uint64>("Dtype"),
    MatMulPtCtOp<uint64, uint64>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulPtCt64").Device(DEVICE_CPU).TypeConstraint<int64>("Dtype"),
    MatMulPtCtOp<int64, uint64>);
