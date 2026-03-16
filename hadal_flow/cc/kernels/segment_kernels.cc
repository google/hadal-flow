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
#include "symmetric_variants.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "utils.h"

using tensorflow::DEVICE_CPU;
using tensorflow::DT_INT32;
using tensorflow::DT_VARIANT;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::int8;
using tensorflow::OkStatus;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::uint64;
using tensorflow::uint8;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

typedef Eigen::ThreadPoolDevice CPUDevice;

// Based on:
// https://github.com/tensorflow/tensorflow/blob/2f084c6f813be21673ecd7bae7a542835cd76637/tensorflow/core/framework/bounds_check.h#L30
// Check that 0 <= index < limit using a single comparison, assuming
// that 0 <= limit if Index is signed.  Intended for use in performance
// critical contexts where 0 <= index < limit is almost always true.
template <typename Ta, typename Tb>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool FastBoundsCheck(Ta const index,
                                                           Tb const limit) {
  static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                "FastBoundsCheck can only be used on integer types.");
  typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
  return TF_PREDICT_TRUE(static_cast<UIndex>(index) <
                         static_cast<UIndex>(limit));
}

// Based on segment_reduction_ops in TensorFlow core with a few changes
// to handle ciphertext batching.
// https://github.com/tensorflow/tensorflow/blob/675237fd0af29df7ebffd9dd2a2f721cd542475b/tensorflow/core/kernels/segment_reduction_ops_impl.h#L353
namespace functor {

template <typename Device, typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor {
  void operator()(OpKernelContext* ctx, ContextVariant<T> const* shell_ctx_var,
                  std::vector<std::shared_ptr<
                      rlwe::RnsGaloisKey<rlwe::MontgomeryInt<T>>>> const& keys,
                  TensorShape const& segment_ids_shape,
                  typename TTypes<Index, 2>::ConstTensor segment_ids,
                  typename TTypes<Variant, 2>::ConstTensor data,
                  typename TTypes<Variant, 2>::Tensor unreduced_output,
                  typename TTypes<Variant, 3>::Tensor output);
};

template <typename T>
struct ZeroCt {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;

  inline SymmetricCtVariant<T> operator()(
      ContextVariant<T> const* shell_ctx_var) const {
    auto zero_generic_ct =
        SymmetricCt::CreateZero(shell_ctx_var->ct_context_->MainPrimeModuli(),
                                shell_ctx_var->error_params_.get());
    SymmetricCt zero_ct(std::move(zero_generic_ct));
    SymmetricCtVariant<T> zero_var(std::move(zero_ct),
                                   shell_ctx_var->ct_context_,
                                   shell_ctx_var->error_params_);
    return zero_var;
  }
};

// The ReductionFunctor implementation for CPU.
// This Op takes two steps:
//    1: Reduce over the ciphertext dimension(s).
//    2: Reduce over the sloting dimension.
// The `unreduced_output` is a temporary tensor which stores intermediate result
// between step 1 and 2.
template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<CPUDevice, T, Index, InitialValueF, ReductionF> {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ShellContext = rlwe::RnsContext<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using Polynomial = rlwe::RnsPolynomial<ModularInt>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;

  void operator()(OpKernelContext* ctx, ContextVariant<T> const* shell_ctx_var,
                  std::vector<std::shared_ptr<RotationKey>> const& keys,
                  TensorShape const& segment_ids_shape,
                  typename TTypes<Index, 2>::ConstTensor segment_ids,
                  typename TTypes<Variant, 2>::ConstTensor data,
                  typename TTypes<Variant, 3>::Tensor output,
                  typename TTypes<Index, 2>::Tensor slot_counter,
                  std::string const& reduction_type) {
    // Initialize the output.
    // auto initial_value = InitialValueF()(shell_ctx_var);
    for (int i = 0; i < output.dimension(0); ++i) {
      for (int ii = 0; ii < output.dimension(1); ++ii) {
        for (int iii = 0; iii < output.dimension(2); ++iii) {
          // output(i, ii, iii) = initial_value;
          output(i, ii, iii) = InitialValueF()(shell_ctx_var);
        }
      }
    }

    if (data.size() == 0) {
      return;
    }

    // Initialize slot counter.
    for (int i = 0; i < slot_counter.dimension(0); ++i) {
      for (int ii = 0; ii < slot_counter.dimension(1); ++ii) {
        slot_counter(i, ii) = 0;
      }
    }

    // Reduce `N` rows input to `num_segments` rows output.
    int64_t const N = segment_ids.dimension(1);
    int64_t const num_segments = output.dimension(1);
    int64_t const inner_dim = data.dimension(1);
    int64_t const num_slots = 1 << shell_ctx_var->log_n_;
    ShellContext const* shell_ctx = shell_ctx_var->ct_context_.get();
    Encoder const* encoder = shell_ctx_var->encoder_.get();
    ReductionF reduction;

    bool trivial_reduction = true;

    // `slot_counter` records which slots in the unreduced_output ciphertext
    // contain real data. This is used to rotate the occupied slots in
    // unreduced_output to the first (and mid) positions for the real output.
    for (int64_t slot = 0; slot < num_slots; ++slot) {
      for (int64_t i = 0; i < N; ++i) {
        Index j = segment_ids(slot, i);
        if (j < 0) {
          continue;
        }
        OP_REQUIRES(
            ctx, FastBoundsCheck(j, num_segments),
            InvalidArgument("segment_ids[", slot, ",", i, "] = ", j,
                            " is out of range [0, ", num_segments, ")"));
        trivial_reduction = false;
        ++slot_counter(slot, j);
      }
    }

    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;

    // Step 1: Reduce over the ciphertext dimension. There are many slots in a
    // ciphertext, and some slots may be assigned to the same output. The
    // `reductionWorker1D` will extract the all slots for the same destination
    // in one mask (multiplication) and one addition (to store the running total
    // over all ciphertexts).
    //
    // Parallelize by `num_segments`. It's simple, efficient and safe
    // (no data dependency):
    //
    //   input   segment_ids                 num_segments  operation
    //   | a0 |  | 0 |            worker 1:  |0|           f(a0, a1)
    //   | b0 |  | 1 |            worker 2:  |1|           f(b0, b1)
    // N | c0 |  | 2 |       -->  worker 3:  |2|           f(c0)
    //   | b1 |  | 1 |
    //   | a1 |  | 0 |
    auto reductionWorker = [&](int64_t begin, int64_t end) -> void {
      // Records which segments IDs have been reduced in a given ciphertext.
      std::unordered_set<int64_t> segment_ids_already_reduced;
      segment_ids_already_reduced.reserve(num_slots);

      for (int64_t i = 0; i < N; ++i) {
        // If this is a new ciphertext, reset which segment IDs were reduced.
        segment_ids_already_reduced.clear();

        for (int64_t slot = 0; slot < num_slots; ++slot) {
          std::vector<uint64_t> mask(num_slots, 0);

          Index j = segment_ids(slot, i);

          // Only act if `j` is in work scope of this worker. Also make sure
          // this segment was not convered by the mask in a previous reduction.
          if (j < begin || j >= end ||
              segment_ids_already_reduced.find(j) !=
                  segment_ids_already_reduced.end()) {
            continue;
          }

          segment_ids_already_reduced.insert(j);

          // Collect a ciphertext-worth of reductions to do in one shot to
          // minimize noise growth. Store which slots are valid in `mask`.
          for (int64_t remaining_slot = slot; remaining_slot < num_slots;
               ++remaining_slot) {
            Index jj = segment_ids(remaining_slot, i);
            mask[remaining_slot] = jj == j;
          }

          for (int64_t chip = 0; chip < inner_dim; ++chip) {
            SymmetricCtVariant<T> const* data_var =
                data(i, chip).get<SymmetricCtVariant<T>>();
            OP_REQUIRES(ctx, data_var != nullptr,
                        InvalidArgument("SymmetricCtVariant for data did not "
                                        "unwrap successfully."));
            OP_REQUIRES_OK(
                ctx,
                const_cast<SymmetricCtVariant<T>*>(data_var)->MaybeLazyDecode(
                    shell_ctx_var->ct_context_, shell_ctx_var->error_params_));

            // Select the desired slots in the ciphertext, masking off the
            // others.
            OP_REQUIRES_VALUE(
                Polynomial mask_pt, ctx,
                encoder->EncodeBgv(mask, shell_ctx->MainPrimeModuli()));
            OP_REQUIRES_VALUE(SymmetricCt masked_data_ct, ctx,
                              data_var->ct * mask_pt);

            SymmetricCtVariant<T>* output_var =
                output(0, (int64_t)j, chip).get<SymmetricCtVariant<T>>();
            OP_REQUIRES(ctx, output_var != nullptr,
                        InvalidArgument("SymmetricCtVariant for output did not "
                                        "unwrap successfully."));
            // No need to lazy decode the output_var, it was created in this op.

            if (output_var->ct.Len() == 0) {
              // Output has not been set yet, wrap the result in a
              // SymmetricCtVariant and store. The output ct will hold raw
              // pointers to moduli stored in the  input's context. Ensure the
              // output ciphertext Variant wrapper holds smart pointers to the
              // input's context to prevent premature deletion of the moduli.
              SymmetricCtVariant var(masked_data_ct, data_var->ct_context,
                                     data_var->error_params);
              output(0, (int64_t)j, chip) = std::move(var);
            } else {
              OP_REQUIRES_OK(ctx, reduction(masked_data_ct, output_var->ct));
            }
          }
        }
      }
    };
    // The reduction (step 1) can be skipped if there is nothing to reduce.
    if (!trivial_reduction) {
      // Use a fixed block size to avoid stragglers in the reduction.
      int64_t const reduction_block_size = 4;
      tsl::thread::ThreadPool::SchedulingParams reduction_scheduling_params(
          tsl::thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
          std::nullopt, reduction_block_size);
      thread_pool->ParallelFor(num_segments, reduction_scheduling_params,
                               reductionWorker);
    }

    // Step 2: Reduce over the slotting dimension. This requires rotating any
    // non-empty slots in the output ciphertexts to the first slot using
    // information stored in the slot counter. Also note, the slotting dimension
    // is not aggregated over the middle slot.
    auto batchAxisReductionWorker = [&](int64_t begin, int64_t end) -> void {
      for (int64_t j = begin; j < end; ++j) {
        for (int64_t chip = 0; chip < inner_dim; ++chip) {
          // Start the reduction for the top and bottom halves of the ciphertext
          // with the output value.
          SymmetricCt accum_top =
              output(0, j, chip).get<SymmetricCtVariant<T>>()->ct;  // deep copy
          SymmetricCt accum_bottom =
              output(0, j, chip).get<SymmetricCtVariant<T>>()->ct;  // deep copy
          // No need to lazy decode the accums, they were created in this op.

          for (int64_t slot = 1; slot < num_slots; ++slot) {
            // Skip the middle slot, it is already included in accum_bottom.
            if (slot == num_slots / 2) continue;

            if (slot_counter(slot, j) > 0) {
              // If this slot was used, rotate slot to first dimension.
              RotationKey const* key;
              int64_t key_slot = slot;
              if (key_slot > num_slots / 2) key_slot = slot - num_slots / 2;
              OP_REQUIRES(ctx, key_slot < static_cast<int64_t>(keys.size()),
                          InvalidArgument("No key for slot '", key_slot, "'"));
              key = keys[key_slot].get();

              SymmetricCt const& ct =
                  output(0, j, chip).get<SymmetricCtVariant<T>>()->ct;

              // Rotate.
              OP_REQUIRES_VALUE(auto ct_sub, ctx,
                                ct.Substitute(key->SubstitutionPower()));
              OP_REQUIRES_VALUE(auto ct_rot, ctx, key->ApplyTo(ct_sub));

              // Call reduce and store in accum.
              if (slot < num_slots / 2) {
                OP_REQUIRES_OK(ctx, reduction(ct_rot, accum_top));
              } else {
                OP_REQUIRES_OK(ctx, reduction(ct_rot, accum_bottom));
              }
            }
          }
          SymmetricCtVariant<T> top_var(std::move(accum_top),
                                        shell_ctx_var->ct_context_,
                                        shell_ctx_var->error_params_);
          SymmetricCtVariant<T> bottom_var(std::move(accum_bottom),
                                           shell_ctx_var->ct_context_,
                                           shell_ctx_var->error_params_);
          output(0, j, chip) = std::move(top_var);
          output(1, j, chip) = std::move(bottom_var);
        }
      }
    };
    if (!trivial_reduction && reduction_type == "galois") {
      // Reduction does not generally have long stragglers, so a variable block
      // size is best to mimiize overhead of parallelization.
      int const cost_per_reduce = 18000 * num_slots;  // ns
      thread_pool->ParallelFor(num_segments, cost_per_reduce,
                               batchAxisReductionWorker);
    }

    // Step 3: For any ciphertexts in the output which are empty, fill in zeros.
    // Create a zero ciphertext by subtracting one of the inputs from itself.
    SymmetricCt zero_ct = data(0, 0).get<SymmetricCtVariant<T>>()->ct;
    OP_REQUIRES_OK(ctx, zero_ct.SubInPlace(zero_ct));
    auto fillInZerosWorker = [&](int64_t begin, int64_t end) -> void {
      for (int64_t i = 0; i < output.dimension(0); ++i) {
        for (int64_t j = begin; j < end; ++j) {
          for (int64_t chip = 0; chip < inner_dim; ++chip) {
            SymmetricCt& ct =
                output(i, j, chip).get<SymmetricCtVariant<T>>()->ct;

            // If the ciphertext degree is zero, it has no moduli and is empty.
            if (ct.Degree() == -1) {
              output(i, j, chip).get<SymmetricCtVariant<T>>()->ct = zero_ct;
            }
          }
        }
      }
    };
    int const cost_per_zero = 900;
    thread_pool->ParallelFor(num_segments, cost_per_zero, fillInZerosWorker);
  }
};

using MatrixChip =
    Eigen::TensorChippingOp<0l, typename TTypes<Variant, 2>::Matrix>;

using constMatrixChip =
    Eigen::TensorChippingOp<0l, typename TTypes<Variant, 2>::ConstMatrix const>;

// reduction functors
template <typename T>
struct SumOp {
  using ModularInt = rlwe::MontgomeryInt<T>;
  using SymmetricCt = rlwe::RnsBgvCiphertext<ModularInt>;

  Status operator()(SymmetricCt const& data, SymmetricCt& output) {
    return output.AddInPlace(data);
  }
};

}  // namespace functor

// Based on:
// https://github.com/tensorflow/tensorflow/blob/d89ba7d5f3701bc468861f845277fc18d1cbc02f/tensorflow/core/kernels/segment_reduction_ops_impl_1.cc#L41
// check routines not in the templated class to reduce code size
template <typename T>
Status ValidateUnsortedSegmentReduction(OpKernel* op_kernel,
                                        OpKernelContext* op_ctx,
                                        ContextVariant<T> const* shell_ctx_var,
                                        Tensor const& data,
                                        Tensor const& segment_ids,
                                        Tensor const& num_segments) {
  if (!TensorShapeUtils::IsScalar(num_segments.shape())) {
    return InvalidArgument("num_segments should be a scalar, not shape ",
                           num_segments.shape().DebugString());
  }

  int64_t num_slots = 1 << shell_ctx_var->log_n_;
  if (segment_ids.dims() == 0 || segment_ids.dim_size(0) != num_slots) {
    return InvalidArgument(
        "segment_ids.shape = ", segment_ids.shape().DebugString(),
        " does not start with number of ciphertext slots = ", num_slots);
  }

  auto segment_ids_suffix_shape = segment_ids.shape();  // copy
  TF_RETURN_IF_ERROR(segment_ids_suffix_shape.RemoveDimWithStatus(0));

  if (!TensorShapeUtils::StartsWith(data.shape(), segment_ids_suffix_shape)) {
    return InvalidArgument("data.shape = ", data.shape().DebugString(),
                           " does not start with segment_ids.shape = ",
                           segment_ids_suffix_shape.DebugString());
  }

  return OkStatus();
}

// The UnsortedSegmentReduction OpKernel. The DeviceReductionFunctor
// is the device specific implementation of the reduction. These device
// specific implementations are templated themselves with the corresponding
// initial value functors and reduction functors.
template <typename T, typename Index, typename DeviceReductionFunctor>
class UnsortedSegmentReductionOp : public OpKernel {
 private:
  using ModularInt = rlwe::MontgomeryInt<T>;
  using ShellContext = rlwe::RnsContext<ModularInt>;
  using RotationKey = rlwe::RnsGaloisKey<ModularInt>;

  std::string reduction_type;
  char const* galois_reduction = "galois";
  char const* no_reduction = "none";

 public:
  explicit UnsortedSegmentReductionOp(OpKernelConstruction* op_ctx)
      : OpKernel(op_ctx), reduction_functor_(DeviceReductionFunctor()) {
    OP_REQUIRES_OK(op_ctx, op_ctx->GetAttr("reduction", &reduction_type));
    OP_REQUIRES(op_ctx, reduction_type == "galois" || reduction_type == "none",
                InvalidArgument("Invalid reduction attribute: ", reduction_type,
                                ". Must be 'galois' or 'none'."));
  }

  void Compute(OpKernelContext* op_ctx) override {
    // Recover the input tensors.
    OP_REQUIRES_VALUE(ContextVariant<T> const* shell_ctx_var, op_ctx,
                      GetVariant<ContextVariant<T>>(op_ctx, 0));
    int64_t const num_slots = 1 << shell_ctx_var->log_n_;
    Tensor const& data = op_ctx->input(1);
    Tensor const& segment_ids = op_ctx->input(2);
    Tensor const& num_segments = op_ctx->input(3);

    // Validate the input tensors.
    OP_REQUIRES_OK(op_ctx, ValidateUnsortedSegmentReduction(
                               this, op_ctx, shell_ctx_var, data, segment_ids,
                               num_segments));

    Index const output_rows = static_cast<Index>(
        num_segments.dtype() == DT_INT32 ? num_segments.scalar<int32>()()
                                         : num_segments.scalar<int64_t>()());
    OP_REQUIRES(op_ctx, output_rows >= 0,
                InvalidArgument("Input num_segments == ", output_rows,
                                " must not be negative."));

    // Recover the rotation keys if needed.
    RotationKeyVariant<T> const* rotation_key_var = nullptr;
    if (reduction_type == galois_reduction) {
      OP_REQUIRES_VALUE(rotation_key_var, op_ctx,
                        GetVariant<RotationKeyVariant<T>>(op_ctx, 4));
      OP_REQUIRES(
          op_ctx, rotation_key_var != nullptr,
          InvalidArgument("RotationKeyVariant did not unwrap successfully."));
      OP_REQUIRES_OK(op_ctx,
                     const_cast<RotationKeyVariant<T>*>(rotation_key_var)
                         ->MaybeLazyDecode(shell_ctx_var->ct_context_));
    }
    std::vector<std::shared_ptr<RotationKey>> empty_rot_keys{};
    std::vector<std::shared_ptr<RotationKey>> const& rot_keys =
        reduction_type == galois_reduction ? rotation_key_var->keys
                                           : empty_rot_keys;

    // Build the output tensor shape.
    TensorShape output_shape;
    OP_REQUIRES_OK(op_ctx, output_shape.AddDimWithStatus(output_rows));
    for (int i = segment_ids.dims() - 1; i < data.dims(); i++) {
      // -1 for batch axis packing.
      OP_REQUIRES_OK(op_ctx, output_shape.AddDimWithStatus(data.dim_size(i)));
    }

    // `slot_counter` records which slots in the output ciphertexts contain real
    // data. This is used to rotate the occupied slots in unreduced_output to
    // the first (and mid) positions. Since this is a plaintext output, prepend
    // the packing dimension. TensorShape slot_counter_shape = output_shape;
    // OP_REQUIRES_OK(op_ctx, slot_counter_shape.InsertDimWithStatus(0,
    // num_slots));
    TensorShape slot_counter_shape = {num_slots, output_rows};
    Tensor* slot_counter = nullptr;
    OP_REQUIRES_OK(
        op_ctx, op_ctx->allocate_output(1, slot_counter_shape, &slot_counter));

    int64_t output_prefix_dim;

    if (reduction_type == galois_reduction) {
      // The reduced output will have a prefix dimension of 2, corresponding to
      // the result from the top and bottom half of the ciphertexts.
      output_prefix_dim = 2;
      OP_REQUIRES_OK(op_ctx,
                     output_shape.InsertDimWithStatus(0, output_prefix_dim));
    } else {
      output_prefix_dim = 1;
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(op_ctx, op_ctx->allocate_output(0, output_shape, &output));

    auto const segment_flat = segment_ids.flat_outer_dims<Index>();
    auto output_flat =
        output->flat_inner_outer_dims<Variant>(output_prefix_dim - 2);
    auto data_flat =
        data.flat_inner_outer_dims<Variant, 2>(segment_ids.dims() - 1 - 1);
    // -1 because flat_inner_outer_dims arg is an includsive range,
    // -1 again for batch axis packing (dimension 0 of data is imaginary).
    auto slot_counter_matrix = slot_counter->flat_outer_dims<Index>();

    reduction_functor_(op_ctx, shell_ctx_var, rot_keys, segment_ids.shape(),
                       segment_flat, data_flat, output_flat,
                       slot_counter_matrix, reduction_type);
  }

 protected:
  DeviceReductionFunctor reduction_functor_;
};

using ShellBaseT = uint64;

REGISTER_KERNEL_BUILDER(
    Name("UnsortedCtSegmentSum")
        .Device(DEVICE_CPU)
        .TypeConstraint<int32>("Tindices"),
    UnsortedSegmentReductionOp<
        ShellBaseT, int32,
        functor::UnsortedSegmentFunctor<CPUDevice, ShellBaseT, int32,
                                        functor::ZeroCt<ShellBaseT>,
                                        functor::SumOp<ShellBaseT>>>);

REGISTER_KERNEL_BUILDER(
    Name("UnsortedCtSegmentSum")
        .Device(DEVICE_CPU)
        .TypeConstraint<int64>("Tindices"),
    UnsortedSegmentReductionOp<
        ShellBaseT, int64,
        functor::UnsortedSegmentFunctor<CPUDevice, ShellBaseT, int64,
                                        functor::ZeroCt<ShellBaseT>,
                                        functor::SumOp<ShellBaseT>>>);
