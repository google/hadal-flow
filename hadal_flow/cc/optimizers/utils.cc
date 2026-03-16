#include "utils.h"

#include "tensorflow/core/framework/node_def.pb.h"

bool IsShellContext(NodeDef const& node) { return node.op() == kShellContext; }
bool IsShellAutoContext(NodeDef const& node) {
  return node.op() == kShellAutoContext;
}

bool IsKeyGen(NodeDef const& node) { return node.op() == kKeyGen; }
bool IsRotationKeyGen(NodeDef const& node) {
  return node.op() == kRotationKeyGen;
}
bool IsFastRotationKeyGen(NodeDef const& node) {
  return node.op() == kFastRotationKeyGen;
}

bool IsModulusReduce(NodeDef const& node) {
  return node.op() == kModulusReduceContext || node.op() == kModulusReduceKey ||
         node.op() == kModulusReduceCt || node.op() == kModulusReducePt;
}

bool IsSaveShellTensor(NodeDef const& node) {
  return node.op() == kSaveShellTensor;
}
bool IsLoadShellTensor(NodeDef const& node) {
  return node.op() == kLoadShellTensor;
}

// Encryption ops.
bool IsEncode(NodeDef const& node) { return node.op() == kEncode; }
bool IsDecode(NodeDef const& node) { return node.op() == kDecode; }
bool IsEncrypt(NodeDef const& node) { return node.op() == kEncrypt; }
bool IsPlainDerypt(NodeDef const& node) { return node.op() == kDecrypt; }
bool IsFastDecrypt(NodeDef const& node) { return node.op() == kFastDecrypt; }
bool IsDecrypt(NodeDef const& node) {
  return IsPlainDerypt(node) || IsFastDecrypt(node);
}

// Arithmetic ops.
bool IsAddCtCt(NodeDef const& node) { return node.op() == kAddCtCt; }
bool IsSubCtCt(NodeDef const& node) { return node.op() == kSubCtCt; }
bool IsMulCtCt(NodeDef const& node) { return node.op() == kMulCtCt; }

bool IsAddCtPt(NodeDef const& node) { return node.op() == kAddCtPt; }
bool IsSubCtPt(NodeDef const& node) { return node.op() == kSubCtPt; }
bool IsMulCtPt(NodeDef const& node) { return node.op() == kMulCtPt; }

bool IsAddPtPt(NodeDef const& node) { return node.op() == kAddPtPt; }
bool IsSubPtPt(NodeDef const& node) { return node.op() == kSubPtPt; }
bool IsMulPtPt(NodeDef const& node) { return node.op() == kMulPtPt; }
bool IsArithmetic(NodeDef const& node) {
  return IsAddCtCt(node) || IsSubCtCt(node) || IsMulCtCt(node) ||
         IsAddCtPt(node) || IsSubCtPt(node) || IsMulCtPt(node) ||
         IsAddPtPt(node) || IsSubPtPt(node) || IsMulPtPt(node);
}

bool IsNegCt(NodeDef const& node) { return node.op() == kNegCt; }
bool IsNegPt(NodeDef const& node) { return node.op() == kNegPt; }

bool IsMulCtTfScalar(NodeDef const& node) {
  return node.op() == kMulCtTfScalar;
}
bool IsMulPtTfScalar(NodeDef const& node) {
  return node.op() == kMulPtTfScalar;
}

// Matrix multiplication ops.
bool IsMatMulCtPt(NodeDef const& node) { return node.op() == kMatMulCtPt; }
bool IsMatMulPtCt(NodeDef const& node) { return node.op() == kMatMulPtCt; }
bool IsFastMatMulPtCt(NodeDef const& node) {
  return node.op() == kFastMatMulPtCt;
}
bool IsTfShellMatMul(NodeDef const& node) {
  return IsMatMulCtPt(node) || IsMatMulPtCt(node) || IsFastMatMulPtCt(node);
}

// Rotation ops.
bool IsRoll(NodeDef const& node) { return node.op() == kRoll; }
bool IsReduceSumByRotation(NodeDef const& node) {
  return node.op() == kReduceSumByRotation;
}
bool IsFastReduceSumByRotation(NodeDef const& node) {
  return node.op() == kFastReduceSumByRotation;
}
bool IsReduceSum(NodeDef const& node) { return node.op() == kReduceSum; }

// Segment sum ops.
bool IsUnsortedCtSegmentSum(NodeDef const& node) {
  return node.op() == kUnsortedCtSegmentSum;
}

// Convolution ops.
bool IsPtCtConv2d(NodeDef const& node) {
  return node.op() == kConv2dPtCt64 || node.op() == kConv2dWithChanPtCt64;
}
bool IsCtPtConv2d(NodeDef const& node) {
  return node.op() == kConv2dCtPt64 || node.op() == kConv2dWithChanCtPt64;
}
bool IsCtCtConv2d(NodeDef const& node) {
  return node.op() == kConv2dCtCt64 || node.op() == kConv2dWithChanCtCt64;
}
bool IsPtCtConv2dTranspose(NodeDef const& node) {
  return node.op() == kConv2dTransposePtCt64 ||
         node.op() == kConv2dTransposeWithChanPtCt64;
}
bool IsCtPtConv2dTranspose(NodeDef const& node) {
  return node.op() == kConv2dTransposeCtPt64 ||
         node.op() == kConv2dTransposeWithChanCtPt64;
}
bool IsCtCtConv2dTranspose(NodeDef const& node) {
  return node.op() == kConv2dTransposeCtCt64 ||
         node.op() == kConv2dTransposeWithChanCtCt64;
}
bool IsPtCtConv2dOrTranspose(NodeDef const& node) {
  return IsPtCtConv2d(node) || IsPtCtConv2dTranspose(node);
}
bool IsCtPtConv2dOrTranspose(NodeDef const& node) {
  return IsCtPtConv2d(node) || IsCtPtConv2dTranspose(node);
}
bool IsCtCtConv2dOrTranspose(NodeDef const& node) {
  return IsCtCtConv2d(node) || IsCtCtConv2dTranspose(node);
}
bool IsConv2d(NodeDef const& node) {
  return IsPtCtConv2dOrTranspose(node) || IsCtPtConv2dOrTranspose(node) ||
         IsCtCtConv2dOrTranspose(node);
}

// Max unpooling ops.
bool IsMaxUnpool2d(NodeDef const& node) {
  return node.op() == kMaxUnpool2dCt64;
}

// hadal shape ops
bool IsConcatCt(NodeDef const& node) { return node.op() == kConcatCt; }

bool IsExpandDimsVariant(NodeDef const& node) {
  return node.op() == kExpandDimsVariant;
}

// TensorFlow ops.
bool IsBroadcastToShape(NodeDef const& node) {
  return node.op() == kBroadcastToShape;
}
bool IsReshape(NodeDef const& node) { return node.op() == kReshape; }
bool IsSplitV(NodeDef const& node) { return node.op() == kSplitVOpName; }

bool NodeOutputsCtOrPt(NodeDef const& node) {
  return IsEncode(node) || IsEncrypt(node) || IsArithmetic(node) ||
         IsNegCt(node) || IsNegPt(node) || IsMulCtTfScalar(node) ||
         IsMulPtTfScalar(node) || IsTfShellMatMul(node) || IsRoll(node) ||
         IsReduceSumByRotation(node) || IsFastReduceSumByRotation(node) ||
         IsReduceSum(node) || IsUnsortedCtSegmentSum(node) || IsConv2d(node) ||
         IsMaxUnpool2d(node) || IsConcatCt(node) || IsModulusReduce(node) ||
         IsExpandDimsVariant(node);
}
