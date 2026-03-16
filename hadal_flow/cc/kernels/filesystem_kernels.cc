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

#include "absl/strings/string_view.h"
#include "context_variant.h"
#include "polynomial_variant.h"
#include "rotation_variants.h"
#include "symmetric_variants.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"

using namespace tensorflow;
using rlwe::MontgomeryInt;

class SaveShellTensorOp : public OpKernel {
 public:
  explicit SaveShellTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor const& input_tensor = ctx->input(0);
    Tensor const& path_tensor = ctx->input(1);

    OP_REQUIRES(ctx, path_tensor.NumElements() == 1,
                absl::InvalidArgumentError("Path must be a scalar"));

    string path = path_tensor.scalar<tstring>()();

    string dir = string(io::Dirname(path));
    absl::Status s = ctx->env()->RecursivelyCreateDir(dir);
    if (!s.ok() && !absl::IsAlreadyExists(s)) {
      OP_REQUIRES_OK(ctx, s);
    }

    std::unique_ptr<WritableFile> file;
    OP_REQUIRES_OK(ctx, ctx->env()->NewWritableFile(path, &file));

    // Write DataType
    DataType dtype = input_tensor.dtype();
    OP_REQUIRES_OK(ctx,
                   file->Append(absl::string_view(
                       reinterpret_cast<char const*>(&dtype), sizeof(dtype))));

    // Write Shape
    TensorShape shape = input_tensor.shape();
    int rank = shape.dims();
    OP_REQUIRES_OK(ctx,
                   file->Append(absl::string_view(
                       reinterpret_cast<char const*>(&rank), sizeof(rank))));
    for (int i = 0; i < rank; ++i) {
      int64_t dim = shape.dim_size(i);
      OP_REQUIRES_OK(ctx,
                     file->Append(absl::string_view(
                         reinterpret_cast<char const*>(&dim), sizeof(dim))));
    }

    if (dtype == DT_VARIANT) {
      auto flat = input_tensor.flat<Variant>();
      int64_t num_elements = flat.size();

      for (int64_t i = 0; i < num_elements; ++i) {
        Variant const& v = flat(i);
        VariantTensorData data;
        v.Encode(&data);

        // Write TypeName
        uint64_t type_name_len = data.type_name_.size();
        OP_REQUIRES_OK(ctx, file->Append(absl::string_view(
                                reinterpret_cast<char const*>(&type_name_len),
                                sizeof(type_name_len))));
        OP_REQUIRES_OK(ctx, file->Append(data.type_name_));

        // Write Metadata
        uint64_t metadata_len = data.metadata_.size();
        OP_REQUIRES_OK(ctx, file->Append(absl::string_view(
                                reinterpret_cast<char const*>(&metadata_len),
                                sizeof(metadata_len))));
        OP_REQUIRES_OK(ctx, file->Append(data.metadata_));

        // Write Tensors
        uint64_t num_tensors = data.tensors_.size();
        OP_REQUIRES_OK(ctx, file->Append(absl::string_view(
                                reinterpret_cast<char const*>(&num_tensors),
                                sizeof(num_tensors))));

        for (Tensor const& t : data.tensors_) {
          TensorProto proto;
          t.AsProtoTensorContent(&proto);
          string s;
          proto.SerializeToString(&s);
          uint64_t s_len = s.size();
          OP_REQUIRES_OK(
              ctx, file->Append(absl::string_view(
                       reinterpret_cast<char const*>(&s_len), sizeof(s_len))));
          OP_REQUIRES_OK(ctx, file->Append(s));
        }
      }
    } else {
      // Fallback for simple types
      OP_REQUIRES_OK(ctx, file->Append(input_tensor.tensor_data()));
    }

    OP_REQUIRES_OK(ctx, file->Close());

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, path_tensor.shape(), &output_tensor));
    output_tensor->flat<tstring>()(0) = path;
  }
};

class LoadShellTensorOp : public OpKernel {
 public:
  explicit LoadShellTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor const& path_tensor = ctx->input(0);
    OP_REQUIRES(ctx, path_tensor.NumElements() == 1,
                absl::InvalidArgumentError("Path must be a scalar"));
    string path = path_tensor.scalar<tstring>()();

    std::unique_ptr<RandomAccessFile> file;
    OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(path, &file));

    uint64_t offset = 0;
    absl::string_view result;
    char scratch[sizeof(uint64_t)];

    // Read DataType
    DataType dtype;
    OP_REQUIRES_OK(ctx, file->Read(offset, sizeof(dtype), &result, scratch));
    std::memcpy(&dtype, result.data(), sizeof(dtype));
    offset += sizeof(dtype);

    // Read Shape
    int rank;
    OP_REQUIRES_OK(ctx, file->Read(offset, sizeof(rank), &result, scratch));
    std::memcpy(&rank, result.data(), sizeof(rank));
    offset += sizeof(rank);

    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; ++i) {
      int64_t dim;
      OP_REQUIRES_OK(ctx, file->Read(offset, sizeof(dim), &result, scratch));
      std::memcpy(&dim, result.data(), sizeof(dim));
      offset += sizeof(dim);
      dims[i] = dim;
    }
    TensorShape shape(dims);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output_tensor));

    OP_REQUIRES(ctx, output_tensor->dtype() == dtype,
                absl::InvalidArgumentError(
                    "File dtype does not match graph expected dtype"));

    if (dtype == DT_VARIANT) {
      auto flat = output_tensor->flat<Variant>();
      int64_t num_elements = flat.size();

      for (int64_t i = 0; i < num_elements; ++i) {
        VariantTensorData data;

        // Read TypeName
        uint64_t type_name_len;
        OP_REQUIRES_OK(
            ctx, file->Read(offset, sizeof(type_name_len), &result, scratch));
        std::memcpy(&type_name_len, result.data(), sizeof(type_name_len));
        offset += sizeof(type_name_len);

        string type_name;
        type_name.resize(type_name_len);

        char* type_name_buf = &type_name[0];
        OP_REQUIRES_OK(
            ctx, file->Read(offset, type_name_len, &result, type_name_buf));
        std::memcpy(type_name_buf, result.data(), type_name_len);
        offset += type_name_len;
        data.type_name_ = type_name;

        // Read Metadata
        uint64_t metadata_len;
        OP_REQUIRES_OK(
            ctx, file->Read(offset, sizeof(metadata_len), &result, scratch));
        std::memcpy(&metadata_len, result.data(), sizeof(metadata_len));
        offset += sizeof(metadata_len);

        string metadata;
        metadata.resize(metadata_len);
        OP_REQUIRES_OK(ctx,
                       file->Read(offset, metadata_len, &result, &metadata[0]));
        std::memcpy(&metadata[0], result.data(), metadata_len);
        offset += metadata_len;
        data.metadata_ = metadata;

        // Read Tensors
        uint64_t num_tensors;
        OP_REQUIRES_OK(
            ctx, file->Read(offset, sizeof(num_tensors), &result, scratch));
        std::memcpy(&num_tensors, result.data(), sizeof(num_tensors));
        offset += sizeof(num_tensors);

        for (uint64_t t_idx = 0; t_idx < num_tensors; ++t_idx) {
          uint64_t s_len;
          OP_REQUIRES_OK(ctx,
                         file->Read(offset, sizeof(s_len), &result, scratch));
          std::memcpy(&s_len, result.data(), sizeof(s_len));
          offset += sizeof(s_len);

          string s;
          s.resize(s_len);

          size_t bytes_read = 0;
          while (bytes_read < s_len) {
            absl::string_view chunk;
            size_t to_read = s_len - bytes_read;
            OP_REQUIRES_OK(ctx,
                           file->Read(offset, to_read, &chunk, &s[bytes_read]));
            if (chunk.size() == 0) {
              OP_REQUIRES(ctx, false, absl::DataLossError("Unexpected EOF"));
            }
            if (chunk.data() != &s[bytes_read]) {
              std::memcpy(&s[bytes_read], chunk.data(), chunk.size());
            }
            bytes_read += chunk.size();
            offset += chunk.size();
          }

          TensorProto proto;
          OP_REQUIRES(
              ctx, proto.ParseFromString(s),
              absl::DataLossError("Failed to parse inner tensor proto"));
          Tensor t;
          OP_REQUIRES(ctx, t.FromProto(proto),
                      absl::DataLossError("Failed to parse inner tensor"));
          data.tensors_.push_back(t);
        }

        // Workaround for potential ABI/Registry issues with Variant::Decode
        Variant v;
        if (data.type_name_ == "ShellSymmetricCtVariant") {
          SymmetricCtVariant<uint64> val;
          val.Decode(data);
          v = std::move(val);
        } else if (data.type_name_ == "ShellPolynomialVariant") {
          PolynomialVariant<uint64> val;
          val.Decode(data);
          v = std::move(val);
        } else if (data.type_name_ == "ShellContextVariant") {
          ContextVariant<uint64> val;
          val.Decode(data);
          v = std::move(val);
        } else if (data.type_name_ == "ShellSymmetricKeyVariant") {
          SymmetricKeyVariant<uint64> val;
          val.Decode(data);
          v = std::move(val);
        } else if (data.type_name_ == "ShellRotationKeyVariant") {
          RotationKeyVariant<uint64> val;
          val.Decode(data);
          v = std::move(val);
        } else if (data.type_name_ == "ShellFastRotationKeyVariant") {
          FastRotationKeyVariant<uint64> val;
          val.Decode(data);
          v = std::move(val);
        } else {
          OP_REQUIRES(ctx, v.Decode(data),
                      absl::DataLossError("Failed to decode variant"));
        }
        flat(i) = std::move(v);
      }
    } else {
      absl::string_view chunk;
      size_t bytes = output_tensor->TotalBytes();
      char* dst = const_cast<char*>(output_tensor->tensor_data().data());

      size_t bytes_read = 0;
      while (bytes_read < bytes) {
        absl::string_view chunk;
        size_t to_read = bytes - bytes_read;
        OP_REQUIRES_OK(ctx,
                       file->Read(offset, to_read, &chunk, dst + bytes_read));
        if (chunk.size() == 0) {
          OP_REQUIRES(ctx, false, absl::DataLossError("Unexpected EOF"));
        }
        if (chunk.data() != dst + bytes_read) {
          std::memcpy(dst + bytes_read, chunk.data(), chunk.size());
        }
        bytes_read += chunk.size();
        offset += chunk.size();
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SaveShellTensor").Device(DEVICE_CPU),
                        SaveShellTensorOp);
REGISTER_KERNEL_BUILDER(Name("LoadShellTensor").Device(DEVICE_CPU),
                        LoadShellTensorOp);