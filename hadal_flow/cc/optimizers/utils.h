#pragma once

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"

using tensorflow::NodeDef;

constexpr char kShellContext[] = "ContextImport64";
constexpr char kShellAutoContext[] = "AutoShellContext64";

constexpr char kEncode[] = "PolynomialImport64";
constexpr char kDecode[] = "PolynomialExport64";
constexpr char kEncrypt[] = "Encrypt64";
constexpr char kDecrypt[] = "Decrypt64";
constexpr char kFastDecrypt[] = "DecryptFastRotated64";

constexpr char kAddCtCt[] = "AddCtCt64";
constexpr char kSubCtCt[] = "SubCtCt64";
constexpr char kMulCtCt[] = "MulCtCt64";

constexpr char kAddCtPt[] = "AddCtPt64";
constexpr char kSubCtPt[] = "SubCtPt64";
constexpr char kMulCtPt[] = "MulCtPt64";

constexpr char kAddPtPt[] = "AddPtPt64";
constexpr char kSubPtPt[] = "SubPtPt64";
constexpr char kMulPtPt[] = "MulPtPt64";

constexpr char kNegPt[] = "NegPt64";
constexpr char kNegCt[] = "NegCt64";

constexpr char kMulCtTfScalar[] = "MulCtTfScalar64";
constexpr char kMulPtTfScalar[] = "MulPtTfScalar64";

constexpr char kMatMulCtPt[] = "MatMulCtPt64";
constexpr char kMatMulPtCt[] = "MatMulPtCt64";
constexpr char kFastMatMulPtCt[] = "FastMatMulPtCt64";

constexpr char kRoll[] = "Roll64";
constexpr char kReduceSumByRotation[] = "ReduceSumByRotationCt64";
constexpr char kFastReduceSumByRotation[] = "FastReduceSumByRotation64";
constexpr char kReduceSum[] = "ReduceSumCt64";

constexpr char kUnsortedCtSegmentSum[] = "UnsortedCtSegmentSum";

constexpr char kConv2dPtCt64[] = "Conv2dPtCt64";
constexpr char kConv2dCtPt64[] = "Conv2dCtPt64";
constexpr char kConv2dCtCt64[] = "Conv2dCtCt64";
constexpr char kConv2dWithChanPtCt64[] = "Conv2dWithChanPtCt64";
constexpr char kConv2dWithChanCtPt64[] = "Conv2dWithChanCtPt64";
constexpr char kConv2dWithChanCtCt64[] = "Conv2dWithChanCtCt64";
constexpr char kConv2dTransposePtCt64[] = "Conv2dTransposePtCt64";
constexpr char kConv2dTransposeCtPt64[] = "Conv2dTransposeCtPt64";
constexpr char kConv2dTransposeCtCt64[] = "Conv2dTransposeCtCt64";
constexpr char kConv2dTransposeWithChanPtCt64[] =
    "Conv2dTransposeWithChanPtCt64";
constexpr char kConv2dTransposeWithChanCtPt64[] =
    "Conv2dTransposeWithChanCtPt64";
constexpr char kConv2dTransposeWithChanCtCt64[] =
    "Conv2dTransposeWithChanCtCt64";

constexpr char kMaxUnpool2dCt64[] = "MaxUnpool2dCt64";

// hadal shape ops
constexpr char kConcatCt[] = "ConcatCt64";
constexpr char kConcatPt[] = "ConcatPt64";

constexpr char kSaveShellTensor[] = "SaveShellTensor";
constexpr char kLoadShellTensor[] = "LoadShellTensor";

constexpr char kKeyGen[] = "KeyGen64";
constexpr char kRotationKeyGen[] = "RotationKeyGen64";
constexpr char kFastRotationKeyGen[] = "FastRotationKeyGen64";

constexpr char kModulusReduceContext[] = "ModulusReduceContext64";
constexpr char kModulusReduceKey[] = "ModulusReduceKey64";
constexpr char kModulusReduceCt[] = "ModulusReduceCt64";
constexpr char kModulusReducePt[] = "ModulusReducePt64";

constexpr char kExpandDimsVariant[] = "ExpandDimsVariant";

// TensorFlow names
constexpr char kBroadcastToShape[] = "BroadcastToShape";  // TODO check name
constexpr char kReshape[] = "Reshape";                    // TODO check name
constexpr char kConstOpName[] = "Const";
constexpr char kSplitVOpName[] = "SplitV";

bool IsShellContext(NodeDef const& node);
bool IsShellAutoContext(NodeDef const& node);
bool IsKeyGen(NodeDef const& node);
bool IsRotationKeyGen(NodeDef const& node);
bool IsFastRotationKeyGen(NodeDef const& node);
bool IsModulusReduce(NodeDef const& node);

bool IsEncode(NodeDef const& node);
bool IsDecode(NodeDef const& node);
bool IsEncrypt(NodeDef const& node);
bool IsPlainDerypt(NodeDef const& node);
bool IsFastDecrypt(NodeDef const& node);
bool IsDecrypt(NodeDef const& node);

bool IsSaveShellTensor(NodeDef const& node);
bool IsLoadShellTensor(NodeDef const& node);

bool IsAddCtCt(NodeDef const& node);
bool IsSubCtCt(NodeDef const& node);
bool IsMulCtCt(NodeDef const& node);

bool IsAddCtPt(NodeDef const& node);
bool IsSubCtPt(NodeDef const& node);
bool IsMulCtPt(NodeDef const& node);

bool IsAddPtPt(NodeDef const& node);
bool IsSubPtPt(NodeDef const& node);
bool IsMulPtPt(NodeDef const& node);
bool IsArithmetic(NodeDef const& node);

bool IsNegCt(NodeDef const& node);
bool IsNegPt(NodeDef const& node);

bool IsMulCtTfScalar(NodeDef const& node);
bool IsMulPtTfScalar(NodeDef const& node);

bool IsMatMulCtPt(NodeDef const& node);
bool IsMatMulPtCt(NodeDef const& node);
bool IsFastMatMulPtCt(NodeDef const& node);
bool IsTfShellMatMul(NodeDef const& node);

bool IsRoll(NodeDef const& node);
bool IsReduceSumByRotation(NodeDef const& node);
bool IsFastReduceSumByRotation(NodeDef const& node);
bool IsReduceSum(NodeDef const& node);

bool IsUnsortedCtSegmentSum(NodeDef const& node);

bool IsPtCtConv2d(NodeDef const& node);
bool IsCtPtConv2d(NodeDef const& node);
bool IsCtCtConv2d(NodeDef const& node);
bool IsPtCtConv2dTranspose(NodeDef const& node);
bool IsCtPtConv2dTranspose(NodeDef const& node);
bool IsCtCtConv2dTranspose(NodeDef const& node);
bool IsPtCtConv2dOrTranspose(NodeDef const& node);
bool IsCtPtConv2dOrTranspose(NodeDef const& node);
bool IsCtCtConv2dOrTranspose(NodeDef const& node);
bool IsConv2d(NodeDef const& node);

bool IsMaxUnpool2d(NodeDef const& node);

bool IsConcatCt(NodeDef const& node);

bool IsExpandDimsVariant(NodeDef const& node);
bool IsBroadcastToShape(NodeDef const& node);
bool IsReshape(NodeDef const& node);
bool IsSplitV(NodeDef const& node);

// Returns true if the custom op node outputs a ciphertext or plaintext.
bool NodeOutputsCtOrPt(NodeDef const& node);
