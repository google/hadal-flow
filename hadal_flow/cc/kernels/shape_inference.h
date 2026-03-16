#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::OkStatus;
using tensorflow::Status;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

Status ImportAndRemoveBatchingDimShape(InferenceContext* c);

template <unsigned int NumOuts>
Status MultiScalarOut(InferenceContext* c) {
  for (unsigned int i = 0; i < NumOuts; i++) {
    c->set_output(i, c->Scalar());
  }
  return OkStatus();
}

template <unsigned int ArgNum>
Status ExportAndAddBatchingDimShape(InferenceContext* c) {
  tsl::int32 batching_dim;
  TF_RETURN_IF_ERROR(c->GetAttr("batching_dim", &batching_dim));

  // If the batching dimension is unknown, set first (packing dimension) of the
  // output shape to unknown.
  ShapeHandle batching_dim_shape;
  if (batching_dim == -1) {
    batching_dim_shape = c->UnknownShapeOfRank(1);
  } else {
    batching_dim_shape = c->MakeShape({batching_dim});
  }

  ShapeHandle output;
  TF_RETURN_IF_ERROR(
      c->Concatenate(batching_dim_shape, c->input(ArgNum), &output));

  c->set_output(0, output);
  return OkStatus();
}

template <unsigned int ArgNum>
Status UnchangedArgShape(InferenceContext* c) {
  c->set_output(0, c->input(ArgNum));
  return OkStatus();
}

Status ShellBroadcastingOpShape(InferenceContext* c);

Status ShellMatMulCtPtShape(InferenceContext* c);

Status ShellMatMulPtCtShape(InferenceContext* c);

Status ShellSegmentReductionWithNumSegmentsShape(InferenceContext* c);

Status ShellConv2d(InferenceContext* c);

Status ShellConv2dWithChan(InferenceContext* c);

Status ShellConv2dTranspose(InferenceContext* c);

Status ShellConv2dTransposeWithChan(InferenceContext* c);

Status ShellSampleCenteredGaussianF(InferenceContext* c);

Status ShellSampleCenteredGaussianL(InferenceContext* c);

Status ShapeFromAttr(InferenceContext* c, char const* attr_name, int output_idx,
                     bool skip_batching_dim = false);