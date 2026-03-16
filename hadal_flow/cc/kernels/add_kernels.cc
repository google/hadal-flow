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
#include "shell_encryption/modulus_conversion.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_bgv_ciphertext.h"
#include "shell_encryption/rns/rns_modulus.h"
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
using tensorflow::uint64;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

// The TensorFlow custom Ops below use the same class for addition and
// subtraction. To do so, they are templated on a functor that performs either
// the addition or subtraction. These functors are defined below.

template <typename OutT, typename LhsT, typename RhsT>
struct ShellAddFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs,
                                            RhsT const& rhs) const {
    return lhs + rhs;
  }
};

template <typename OutT, typename LhsT, typename RhsT>
struct ShellSubFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs,
                                            RhsT const& rhs) const {
    return lhs - rhs;
  }
};

template <typename LhsT, typename RhsT>
struct ShellAddInPlaceFunctor {
  constexpr absl::Status operator()(LhsT& lhs, RhsT const& rhs) const {
    return lhs.AddInPlace(rhs);
  }
};

template <typename LhsT, typename RhsT>
struct ShellSubInPlaceFunctor {
  constexpr absl::Status operator()(LhsT& lhs, RhsT const& rhs) const {
    return lhs.SubInPlace(rhs);
  }
};

template <typename OutT, typename LhsT, typename RhsT, typename ModParamsT>
struct ShellAddWithParamsFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs, RhsT const& rhs,
                                            ModParamsT const& p) const {
    return lhs.Add(rhs, p);
  }
};

template <typename OutT, typename LhsT, typename RhsT, typename ModParamsT>
struct ShellSubWithParamsFunctor {
  constexpr rlwe::StatusOr<OutT> operator()(LhsT const& lhs, RhsT const& rhs,
                                            ModParamsT const& p) const {
    return lhs.Sub(rhs, p);
  }
};

template <typename T, typename ShellAddSub>
class AddCtCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit AddCtCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

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

    // Allocate the output tensor which is the same size as one of the inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    auto add_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<T> const* ct_a_var =
            flat_a(a_bcaster(i)).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(
            op_ctx, ct_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index: ", i,
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
            InvalidArgument("SymmetricCtVariant at flat index: ", i,
                            " for input b did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_b_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct_b = ct_b_var->ct;

        ShellAddSub add_or_sub;
        OP_REQUIRES_VALUE(SymmetricCt ct_c, op_ctx, add_or_sub(ct_a, ct_b));

        // SHELL's addition preserves moduli pointers of the first input.
        // Ensure the output holds smart pointers to the input's context to
        // prevent premature deletion of the moduli.
        SymmetricCtVariant ct_c_var(std::move(ct_c), ct_a_var->ct_context,
                                    ct_a_var->error_params);
        flat_output(i) = std::move(ct_c_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_add = 30 * num_slots * num_components;
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_add,
                             add_in_range);
  }
};

template <typename T, typename ShellAddSub>
class AddCtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit AddCtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

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

    // Allocate the output tensor which is the same size as one of the inputs.
    Tensor* output;
    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));
    auto flat_output = output->flat<Variant>();

    auto add_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<T> const* ct_a_var =
            flat_a(a_bcaster(i)).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(
            op_ctx, ct_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index: ", i,
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
            InvalidArgument("PolynomialVariant at flat index: ", i,
                            " for input b did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<PolynomialVariant<T>*>(pv_b_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_));
        RnsPolynomial const& pt_b = pv_b_var->poly;

        ShellAddSub add_or_sub;
        OP_REQUIRES_VALUE(SymmetricCt ct_c, op_ctx, add_or_sub(ct_a, pt_b));

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
    int const cost_per_add = 30 * num_slots * num_components;
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_add,
                             add_in_range);
  }
};

template <typename T, typename ShellAddSubWithParams>
class AddPtPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit AddPtPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
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

    // Allocate the output tensor which is the same size as one of the inputs.
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
            InvalidArgument("PolynomialVariant at flat index: ", i,
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
            InvalidArgument("PolynomialVariant at flat index: ", i,
                            " for input b did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<PolynomialVariant<T>*>(pv_b_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_));
        RnsPolynomial const& pt_b = pv_b_var->poly;

        ShellAddSubWithParams add_or_sub;
        OP_REQUIRES_VALUE(RnsPolynomial pt_c, op_ctx,
                          add_or_sub(pt_a, pt_b, shell_ctx->MainPrimeModuli()));

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

template <typename T>
class NegCtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ModularIntParams = typename ModularInt::Params;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

 public:
  explicit NegCtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Tensor const& a = op_ctx->input(1);
    auto flat_a = a.flat<Variant>();

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

    // Allocate the output tensor which is the same size as the input.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    // Set up flat view of the output tensor.
    auto flat_output = output->flat<Variant>();

    auto negate_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        SymmetricCtVariant<T> const* ct_a_var =
            flat_a(i).get<SymmetricCtVariant<T>>();
        OP_REQUIRES(
            op_ctx, ct_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index: ", i,
                            " for input a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<SymmetricCtVariant<T>*>(ct_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_, shell_ctx_var->error_params_));
        SymmetricCt const& ct_a = ct_a_var->ct;

        OP_REQUIRES_VALUE(auto ct_out, op_ctx, ct_a.Negate());

        // The output ct will hold smart pointers to the input's context
        // to prevent premature deletion of the moduli.
        SymmetricCtVariant ct_out_var(std::move(ct_out), ct_a_var->ct_context,
                                      ct_a_var->error_params);
        flat_output(i) = std::move(ct_out_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_neg = 20 * num_slots * num_components;
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_neg,
                             negate_in_range);
  }
};

template <typename T>
class NegPtOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using Context = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;

 public:
  explicit NegPtOp(OpKernelConstruction* op_ctx) : OpKernel(op_ctx) {}

  void Compute(OpKernelContext* op_ctx) override {
    // Unpack the input arguments.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    Context const* shell_ctx = shell_ctx_var->ct_context_.get();

    Tensor const& a = op_ctx->input(1);
    auto flat_a = a.flat<Variant>();

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

    // Allocate the output tensor which is the same size as the input.
    Tensor* output;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, a.shape(), &output));

    // Set up flat view of the output tensor.
    auto flat_output = output->flat<Variant>();

    auto negate_in_range = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        PolynomialVariant<T> const* pt_a_var =
            flat_a(i).get<PolynomialVariant<T>>();
        OP_REQUIRES(
            op_ctx, pt_a_var != nullptr,
            InvalidArgument("SymmetricCtVariant at flat index: ", i,
                            " for input a did not unwrap successfully."));
        OP_REQUIRES_OK(
            op_ctx,
            const_cast<PolynomialVariant<T>*>(pt_a_var)->MaybeLazyDecode(
                shell_ctx_var->ct_context_));
        RnsPolynomial const& pt_a = pt_a_var->poly;

        OP_REQUIRES_VALUE(RnsPolynomial pt_out, op_ctx,
                          pt_a.Negate(shell_ctx->MainPrimeModuli()));
        PolynomialVariant pt_out_var(std::move(pt_out),
                                     shell_ctx_var->ct_context_);
        flat_output(i) = std::move(pt_out_var);
      }
    };

    auto thread_pool =
        op_ctx->device()->tensorflow_cpu_worker_threads()->workers;
    int const cost_per_neg = 20 * num_slots;
    thread_pool->ParallelFor(flat_output.dimension(0), cost_per_neg,
                             negate_in_range);
  }
};

using BaseInt64 = uint64;
using ModularInt64 = rlwe::MontgomeryInt<BaseInt64>;
// using ModularIntParams64 = typename rlwe::MontgomeryInt<BaseInt64>::Params;
using Moduli = absl::Span<rlwe::PrimeModulus<ModularInt64> const* const>;
using RnsPolynomial64 = rlwe::RnsPolynomial<ModularInt64>;
using SymmetricCt64 = rlwe::RnsBgvCiphertext<ModularInt64>;

REGISTER_KERNEL_BUILDER(
    Name("AddCtCt64").Device(DEVICE_CPU),
    AddCtCtOp<BaseInt64,
              ShellAddFunctor<SymmetricCt64, SymmetricCt64, SymmetricCt64>>);

REGISTER_KERNEL_BUILDER(
    Name("AddCtPt64").Device(DEVICE_CPU),
    AddCtPtOp<BaseInt64,
              ShellAddFunctor<SymmetricCt64, SymmetricCt64, RnsPolynomial64>>);

REGISTER_KERNEL_BUILDER(
    Name("AddPtPt64").Device(DEVICE_CPU),
    AddPtPtOp<BaseInt64,
              ShellAddWithParamsFunctor<RnsPolynomial64, RnsPolynomial64,
                                        RnsPolynomial64, Moduli>>);

REGISTER_KERNEL_BUILDER(
    Name("SubCtCt64").Device(DEVICE_CPU),
    AddCtCtOp<BaseInt64,
              ShellSubFunctor<SymmetricCt64, SymmetricCt64, SymmetricCt64>>);

REGISTER_KERNEL_BUILDER(
    Name("SubCtPt64").Device(DEVICE_CPU),
    AddCtPtOp<BaseInt64,
              ShellSubFunctor<SymmetricCt64, SymmetricCt64, RnsPolynomial64>>);

REGISTER_KERNEL_BUILDER(
    Name("SubPtPt64").Device(DEVICE_CPU),
    AddPtPtOp<BaseInt64,
              ShellSubWithParamsFunctor<RnsPolynomial64, RnsPolynomial64,
                                        RnsPolynomial64, Moduli>>);

REGISTER_KERNEL_BUILDER(Name("NegCt64").Device(DEVICE_CPU), NegCtOp<BaseInt64>);
REGISTER_KERNEL_BUILDER(Name("NegPt64").Device(DEVICE_CPU), NegPtOp<BaseInt64>);
