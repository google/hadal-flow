#include "moduli_autotune.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "shell_encryption/rns/rns_error_params.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr bool const qs_mod_t_is_one = true;

constexpr bool const debug_moduli = false;
constexpr bool const debug_noise_estimation = false;
constexpr bool const debug_graph = false;
constexpr bool const debug_output_params = true;

constexpr uint64_t const kMaxPrimeBitsPlaintext = 58;
constexpr uint64_t const kMaxPrimeBitsCiphertext = 60;
constexpr uint64_t const kMinPrimeBits = 3;

struct ShellParams {
  uint64_t log_n;
  uint64_t t;
  std::vector<uint64_t> qs;
};

struct ShellAutoParams {
  uint64_t cleartext_bits;
  float scaling_factor;
  int64_t noise_offset_bits;
  uint64_t noise_variance;
  std::string seed;
};

// This function returns the uint64_t value of a scalar constant node or an
// error if this node is not a constant, scalar, or the wrong type. It is based
// on TensorFlow's GetScalarConstNodeValueHelper() in graph_utils.cc.
template <typename T, int TF_T>
Status GetScalarConstValue(NodeDef const& node, T* value) {
  if (node.op() != kConstOpName)
    return errors::InvalidArgument("Node ", node.name(),
                                   " is not a Const node. Op: ", node.op());

  Tensor tensor;
  TF_RETURN_IF_ERROR(GetNodeAttr(node, "value", &tensor));
  if (!TensorShapeUtils::IsScalar(tensor.shape())) {
    return errors::InvalidArgument(
        "Node ", node.name(),
        " should be a scalar but has shape: ", tensor.shape());
  }

  if (tensor.dtype() != TF_T) {
    return errors::InvalidArgument(
        "Node ", node.name(), " should have type ", DataTypeString(DT_UINT64),
        " but has type: ", DataTypeString(tensor.dtype()));
  }

  *value = tensor.scalar<T>()();

  return absl::OkStatus();
}

template <typename T>
Status AddScalarConstNode(T value, utils::Mutation* mutation,
                          std::string const& name, std::string const& device) {
  NodeDef node;
  node.set_op(kConstOpName);
  node.set_name(name);
  node.set_device(device);

  std::unique_ptr<tensorflow::TensorProto> tensor =
      std::make_unique<tensorflow::TensorProto>();
  std::unique_ptr<tensorflow::TensorShapeProto> tensor_shape =
      std::make_unique<tensorflow::TensorShapeProto>();

  if constexpr (std::is_same<T, uint64_t>::value) {
    (*node.mutable_attr())["dtype"].set_type(DT_UINT64);
    tensor->set_dtype(DT_UINT64);
    tensor->add_uint64_val(value);
  } else if constexpr (std::is_same<T, std::string>::value) {
    (*node.mutable_attr())["dtype"].set_type(DT_STRING);
    tensor->set_dtype(DT_STRING);
    tensor->add_string_val(value);
  } else if constexpr (std::is_same<T, std::vector<uint64_t>>::value) {
    (*node.mutable_attr())["dtype"].set_type(DT_UINT64);
    tensor->set_dtype(DT_UINT64);
    tensor_shape->add_dim()->set_size(value.size());
    for (auto const& v : value) {
      tensor->add_uint64_val(v);
    }
  } else {
    // clang-format off
    []<bool flag = false>() {
      static_assert(flag, "AddScalarConstNode does not support this type");

    }();
    // clang-format on
  }
  tensor->set_allocated_tensor_shape(tensor_shape.release());
  (*node.mutable_attr())["value"].set_allocated_tensor(tensor.release());

  if constexpr (debug_graph) {
    std::cout << "Adding scalar const node: " << node.DebugString()
              << std::endl;
  }

  Status status;
  mutation->AddNode(std::move(node), &status);
  return status;
}

utils::MutableNodeView* GetNextAutoShellContextNode(
    utils::MutableGraphView& graph_view) {
  for (int i = 0; i < graph_view.NumNodes(); ++i) {
    auto const* node_view = graph_view.GetNode(i);
    auto const* node_def = node_view->node();
    if (node_def->op() == kShellAutoContext) {
      return graph_view.GetNode(i);
    }
  }
  return nullptr;
}

Status GetAutoShellContextParams(utils::MutableNodeView* autocontext,
                                 ShellAutoParams& params) {
  auto const* cleartext_bits_node =
      autocontext->GetRegularFanin(0).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<uint64_t, DT_UINT64>(
      *cleartext_bits_node, &params.cleartext_bits));

  auto const* scaling_factor_node =
      autocontext->GetRegularFanin(1).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<float, DT_FLOAT>(
      *scaling_factor_node, &params.scaling_factor));

  auto const* noise_offset_node =
      autocontext->GetRegularFanin(2).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<int64_t, DT_INT64>(
      *noise_offset_node, &params.noise_offset_bits));

  auto const* noise_variance_node =
      autocontext->GetRegularFanin(3).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<uint64_t, DT_UINT64>(
      *noise_variance_node, &params.noise_variance));

  if constexpr (debug_moduli) {
    std::cout << "Cleartext Bits: " << params.cleartext_bits << std::endl;
    std::cout << "Scaling Factor: " << params.scaling_factor << std::endl;
    std::cout << "Noise Offset Bits: " << params.noise_offset_bits << std::endl;
    std::cout << "Noise Variance: " << params.noise_variance << std::endl;
  }

  return OkStatus();
}

StatusOr<bool> DecryptUsesSameContext(utils::MutableNodeView const* node_view,
                                      utils::MutableNodeView const* context) {
  utils::MutableNodeView const* trace = node_view;
  if (trace == nullptr ||
      !(IsDecrypt(*trace->node()) || IsDecode(*trace->node()))) {
    return errors::InvalidArgument(
        "Expected the node to be a decrypt node, but found ", trace->GetOp());
  }
  trace = trace->GetRegularFanin(0).node_view();

  // The next op should be a strided slice.
  if (trace->GetOp() != "StridedSlice") {
    return errors::InvalidArgument(
        "Traceback to context expected the first op to be a strided slice, "
        "but found ",
        trace->GetOp());
  }
  trace = trace->GetRegularFanin(0).node_view();

  // The next op could be a tensor list gather (if the context was created on
  // a local device) or a ParseTensor (if the context is read from a cache).
  if (trace->GetOp() != "TensorListGather" && trace->GetOp() != "ParseTensor") {
    std::cout << "Trace: " << trace->node()->DebugString() << std::endl;
    return errors::InvalidArgument(
        "Traceback to context expected the second op to be a tensor list "
        "gather, but found ",
        trace->GetOp());
  }

  // Tracing further back in the graph is difficult because of how TensorFlow
  // decides to optimize the graph before this optimizer is run. It is
  // difficult because the context may be cached and read from disk.
  // Instead of handling all possible cases, take advantage of the name scope.
  // The context part of the name scopes of the TensorListGather should
  // match that of the context node. This is more fragile on the hadal
  // side, but will not break if the TensorFlow graph optimizers change.
  std::string actx_name = context->GetName();
  int actx_ns_start = actx_name.find("create_autocontext64");
  int actx_ns_end = actx_name.find("/", actx_ns_start);
  std::string actx_ns =
      actx_name.substr(actx_ns_start, actx_ns_end - actx_ns_start);

  if (trace->GetName().find(actx_ns) == std::string::npos) {
    return false;
  }
  return true;
}

StatusOr<float> MaxScalingFactor(utils::MutableGraphView& graph_view,
                                 utils::MutableNodeView const* autocontext) {
  // Traverse the graph and return the maximum scaling factor across all decrypt
  // ops tied to this autocontext.
  int const num_nodes = graph_view.NumNodes();
  float max_sf = 0;

  for (int i = 0; i < num_nodes; ++i) {
    auto const* this_node_view = graph_view.GetNode(i);
    auto const* this_node_def = this_node_view->node();

    // Decryption is where the maximum multiplicative depth is reached.
    if (IsDecrypt(*this_node_def) || IsDecode(*this_node_def)) {
      // Ensure the decrypt op uses the same autocontext node as the argument
      // (for the case where there are multiple autocontext nodes in the graph).
      TF_ASSIGN_OR_RETURN(bool is_same_autocontext,
                          DecryptUsesSameContext(this_node_view, autocontext));

      float sf = 0.;
      if (!TryGetNodeAttr(*this_node_def, "scaling_factor", &sf)) {
        std::cout
            << "WARNING: Could not determine scaling factor in Decrypt op."
            << " Plaintext modulus may be underprovisioned." << std::endl;
      }

      if (is_same_autocontext) {
        max_sf = std::max(max_sf, sf);
      }
    }
  }
  return max_sf;
}

// Function for modular exponentiation
uint64_t modPow(uint64_t base, uint64_t exp, uint64_t modulus) {
  typedef unsigned __int128 uint128_t;
  base %= modulus;
  uint128_t result = 1;
  while (exp > 0) {
    if (exp & 1) result = (result * base) % modulus;
    base = (uint128_t(base) * base) % modulus;
    exp >>= 1;
  }
  return result;
}

// Fermat's prime test.
bool CheckPrime(uint64_t const n, uint64_t const k, std::mt19937& engine) {
  if (n == 2) return true;
  if ((n & 1) == 0) return false;

  std::uniform_int_distribution<uint64_t> dist(2, n - 2);

  for (uint64_t i = 0; i < k; ++i) {
    uint64_t a = dist(engine);
    if (modPow(a, n - 1, n) != 1) return false;
  }
  return true;
}

uint64_t FindPrimeMod2n(uint64_t const two_n, uint64_t const bits_start,
                        uint64_t const bits_end,
                        std::vector<uint64_t> const& qs = {},
                        uint64_t const t = 0) {
  // Prepare for random number generation.
  std::mt19937 engine(static_cast<long unsigned int>(std::time(nullptr)));

  uint64_t const start = uint64_t(1) << bits_start;
  uint64_t const end = uint64_t(1) << bits_end;

  // Handle the case when end is larger than start, iterate in reverse.
  bool const reverse = start > end;

  if constexpr (debug_moduli) {
    std::cout << "Finding prime between " << bits_start << " and " << bits_end
              << " bits which is congruent to 1 mod 2n (" << two_n << ")."
              << std::endl;
  }

  for (uint64_t i = start; reverse ? i > end : i < end; reverse ? --i : ++i) {
    // Check i mod t is 1.
    if constexpr (qs_mod_t_is_one) {
      if (t != 0) {
        uint64_t i_mod_t = i % t;
        if (i_mod_t != 1) {
          // Given i mod t is not 1, skip ahead.
          if (i_mod_t != 0 && i > t) {
            if (reverse) {
              i -= (i_mod_t - 2);
            } else {
              i += t - i_mod_t;
            }
          }
          continue;
        }
      }
    }

    // Check if i is congruent to 1 mod 2n.
    if (i % two_n != 1) continue;

    // Check if i is in the qs list.
    if (std::find(qs.begin(), qs.end(), i) != qs.end()) continue;

    // Check if i is prime. This is the most computationally expensive test so
    // perform it last.
    if (!CheckPrime(i, 10, engine)) continue;
    return i;
  }
  return 0;
}

constexpr uint64_t BitWidth(uint64_t n) {
  uint64_t bits = 0;
  while (n) {
    n >>= 1;
    ++bits;
  }
  return bits;
}

// Choose primes for the RNS ciphertext moduli chain. The primes are all
// congruent to 1 mod 2n and 1 mod t. The number of bits in the product of
// the primes returned is at least log_q. The aggression parameter controls
// how optimistic the search expects to find primes at the upper end of the
// range. E.g. if aggression is 2, the algorithm will assume that a prime can
// be found at the upper end of the range with 2 fewer bits than the maximum.
// This is useful to minimize the number of primes in the chain.
std::vector<uint64_t> ChooseRnsCtModuli(uint64_t const two_n, uint64_t const t,
                                        uint64_t const log_q,
                                        uint64_t const aggression) {
  std::vector<uint64_t> qs;
  int64_t needed_bits_of_primes = log_q;
  bool is_first_prime = true;

  // Find prime numbers for the RNS chain of ciphertext moduli.
  while (needed_bits_of_primes > 0) {
    uint64_t range_start;
    uint64_t range_end;
    uint64_t smallest_prime;

    // Check how many 64-bit primes are left to find.
    if (needed_bits_of_primes >
        2 * static_cast<int64_t>(kMaxPrimeBitsCiphertext - aggression)) {
      // More than two primes are left to find. The goal is to find the largest
      // prime possible.
      //
      // The first prime is special and must be larger than the plaintext
      // modulus. This is because decryption ModReduces to the first RNS prime,
      // then to the plaintext modulus. For remaining moduli, this restriction
      // is not necessary (assuming qs_mod_t_is_one is false) and the full range
      // may be searched.
      //
      // Since subsequent primes are needed after this one, start the search
      // from the end of the range to find the largest one.
      range_start = kMaxPrimeBitsCiphertext;
      range_end = is_first_prime ? BitWidth(t) + 1 : kMinPrimeBits;
    } else if (needed_bits_of_primes >
               static_cast<int64_t>(kMaxPrimeBitsCiphertext - aggression)) {
      // Only two primes are left to find. Instead of finding the largest prime
      // possible for this second to last position, leaving potentially too few
      // bits for the next prime, try to balance the bits between the two
      // primes. To perfectly balance the primes, the goal is for each to have
      // half the number of required bits, i.e. needed_bits_of_primes / 2.
      //
      // Search from the beginning of the range to find the smallest prime.
      range_start = static_cast<uint64_t>(needed_bits_of_primes / 2);

      // As above, the first prime is special and must be larger than the
      // plaintext modulus.
      if (is_first_prime) {
        range_start = std::max(BitWidth(t) + 1, range_start);
      }

      range_end = kMaxPrimeBitsCiphertext;

    } else {
      // If only one prime more prime is needed, start the search from the
      // begining of the range to find the smallest one.
      range_start = needed_bits_of_primes;
      range_end = kMaxPrimeBitsCiphertext;
    }

    if constexpr (qs_mod_t_is_one) {
      // When each q_i is required to be congruent to 1 mod t, all primes must
      // be larger than t.
      if (range_start < range_end) {
        range_start = std::max(BitWidth(t) + 1, range_start);
      } else {
        range_end = std::max(BitWidth(t) + 1, range_end);
      }
    }

    uint64_t prime = FindPrimeMod2n(two_n, range_start, range_end, qs, t);

    if (prime == 0) {
      static bool warning_printed = false;
      if (!warning_printed) {
        std::cout << "ERROR: Could not find a prime for RNS ct prime between "
                  << range_start << " and " << range_end
                  << " bits which is congruent to 1 mod 2n (" << two_n
                  << ") and congruent to 1 mod t (" << t << ")." << std::endl;
        warning_printed = true;
        return {};
      }
    }
    qs.push_back(prime);
    needed_bits_of_primes -= BitWidth(prime);
    is_first_prime = false;
  }

  return qs;
}

uint64_t EstimateLogN(uint64_t log_q) {
  // Values from standard v1.1 table 1 for 128 bits of security.
  // @techreport{HomomorphicEncryptionSecurityStandard,
  // author = {Martin Albrecht and Melissa Chase and Hao Chen and Jintai Ding
  // and Shafi Goldwasser and Sergey Gorbunov and Shai Halevi and Jeffrey
  // Hoffstein and Kim Laine and Kristin Lauter and Satya Lokam and Daniele
  // Micciancio and Dustin Moody and Travis Morrison and Amit Sahai and Vinod
  // Vaikuntanathan}, title = {Homomorphic Encryption Security Standard},
  // institution= {HomomorphicEncryption.org},
  // publisher = {HomomorphicEncryption.org},
  // address = {Toronto, Canada},
  // year = {2018},
  // month = {November}
  // }
  if (log_q <= 29) return 10;
  if (log_q <= 56) return 11;
  if (log_q <= 111) return 12;
  if (log_q <= 220) return 13;
  if (log_q <= 440) return 14;
  if (log_q <= 880) return 15;
  return 0;
}

Status ChooseShellParams(ShellParams& params, uint64_t const total_pt_bits,
                         uint64_t total_ct_bits) {
  // Estimate log_n from the needed number of ct bits.
  uint64_t log_n = EstimateLogN(total_ct_bits);
  if (log_n == 0) {
    return errors::FailedPrecondition("Could not estimate log_n.");
  }
  uint64_t two_n = 1 << (log_n + 1);

  uint64_t bounded_total_pt_bits = std::max(total_pt_bits, kMinPrimeBits);

  uint64_t t =
      FindPrimeMod2n(two_n, bounded_total_pt_bits, kMaxPrimeBitsPlaintext);
  if (t == 0) {
    static bool warning_printed = false;
    if (!warning_printed) {
      std::cout
          << "ERROR: Could not find a prime for plaintext modulus between "
          << bounded_total_pt_bits << " and " << kMaxPrimeBitsPlaintext
          << " bits." << std::endl;
      warning_printed = true;
    }
    return errors::FailedPrecondition(
        "Could not find a prime for plaintext modulus.");
  }

  std::vector<uint64_t> qs;
  uint64_t aggression = 1;
  while (qs.empty() && aggression < 8) {
    if constexpr (debug_moduli) {
      std::cout << "Searching for " << total_ct_bits
                << " bits of qs with aggression parameter " << aggression
                << std::endl;
    }
    qs = ChooseRnsCtModuli(two_n, t, total_ct_bits, aggression);
    aggression++;
  }
  if (qs.empty()) {
    std::cout << "ERROR: Could not find prime(s) for ciphertext modulus."
              << std::endl;
    return errors::FailedPrecondition(
        "Could not find prime(s) for ciphertext modulus.");
  }

  // Since the ciphertext bits may be larger than the minimum, update logn.
  // This estimation counts the number of bits conservatively.
  uint64_t found_ct_bits = 0;
  for (auto const& q : qs) {
    found_ct_bits += BitWidth(q) - 1;
  }
  uint64_t new_log_n = EstimateLogN(found_ct_bits);
  if (new_log_n == 0) {
    return errors::FailedPrecondition("Could not estimate log_n.");
  }
  if (new_log_n != log_n) {
    // log_n has changed, all parameters must be updated.
    log_n = new_log_n;
    return ChooseShellParams(params, total_pt_bits, found_ct_bits);
  }

  params.log_n = log_n;
  params.t = t;
  params.qs = std::move(qs);

  if constexpr (debug_moduli) {
    std::cout << "Found parameters:" << std::endl;
    std::cout << "  log_n: " << params.log_n << std::endl;
    std::cout << "  t: " << params.t << std::endl;
    std::cout << "  qs: ";
    for (auto const& q : params.qs) {
      std::cout << q << " ";
    }
    std::cout << std::endl;
  }
  return OkStatus();
}

// Returns the noise budget of the current node.
template <typename T>
Status EstimateNodeNoise(
    utils::MutableGraphView& graph_view, int node_index,
    std::vector<uint64_t>& node_noise, ShellParams const& params,
    rlwe::RnsErrorParams<rlwe::MontgomeryInt<T>>& error_params) {
  // Get the current node and its fanin nodes.
  auto const* node_view = graph_view.GetNode(node_index);
  auto const* node_def = node_view->node();

  uint64_t* this_noise = &node_noise[node_index];
  uint64_t noise_a = 0;
  uint64_t noise_b = 0;
  if (node_view->GetRegularFanin(1).node_index() < node_noise.size()) {
    noise_a = node_noise[node_view->GetRegularFanin(1).node_index()];
  }
  if (node_view->GetRegularFanin(2).node_index() < node_noise.size()) {
    noise_b = node_noise[node_view->GetRegularFanin(2).node_index()];
  }

  // Setup constants for estimating rotation noise.
  // TODO: each of these should be calculated based on the actual parameters.
  int const kNumComponents =
      2;  // TODO need to get num components based on ct*ct depth.
  constexpr int kLogGadgetBase = 4;  // TODO from rotation_kernels.cc
  int gadget_dimension = 0;
  for (auto const& q : params.qs)
    gadget_dimension += (BitWidth(q) + (kLogGadgetBase - 1)) / kLogGadgetBase;

  if (IsEncrypt(*node_def)) {
    *this_noise = BitWidth(error_params.B_secretkey_encryption());
  }

  // CtCt operations.
  else if (IsAddCtCt(*node_def) || IsSubCtCt(*node_def)) {
    *this_noise = std::max(noise_a, noise_b) + 1;
  } else if (IsMulCtCt(*node_def)) {
    *this_noise = noise_a + noise_b;
  }

  // CtPt operations.
  else if (IsAddCtPt(*node_def) || IsSubCtPt(*node_def)) {
    *this_noise = noise_a;
  } else if (IsMulCtPt(*node_def)) {
    *this_noise = noise_a + BitWidth(error_params.B_plaintext());
  }

  // Negation operations.
  else if (IsNegCt(*node_def)) {
    *this_noise = noise_a;
  }

  // Scalar multiplication operations.
  else if (IsMulCtTfScalar(*node_def)) {
    auto scalar_node_view = node_view->GetRegularFanin(2).node_view();
    auto const* scalar_node_def = scalar_node_view->node();

    // Otherwise this is a normal multiplication
    Status s;
    uint64_t scalar_uint64 = 0;
    s = GetScalarConstValue<uint64_t, DT_UINT64>(*scalar_node_def,
                                                 &scalar_uint64);
    if (!s.ok()) {
      int64_t scalar_int64 = 0;
      s = GetScalarConstValue<int64_t, DT_INT64>(*scalar_node_def,
                                                 &scalar_int64);
      if (!s.ok()) {
        // If there was a cast in the chain of ops traversed above, input
        // might be a float.
        float scalar_float = 0;
        s = GetScalarConstValue<float, DT_FLOAT>(*scalar_node_def,
                                                 &scalar_float);
        if (!s.ok()) {
          double scalar_double = 0;
          s = GetScalarConstValue<double, DT_DOUBLE>(*scalar_node_def,
                                                     &scalar_double);
          if (!s.ok()) {
            static bool warning_printed = false;
            if (!warning_printed) {
              std::cout << "WARNING: Moduli autotune could not determine size "
                           "of MulCtTf plaintext operand. Noise budget may be "
                           "underestimated."
                        << std::endl;
              warning_printed = true;
            }
            *this_noise = noise_a;
          } else {
            *this_noise =
                noise_a + BitWidth(static_cast<int64_t>(scalar_double));
          }
        } else {
          *this_noise = noise_a + BitWidth(static_cast<int64_t>(scalar_float));
        }
      } else {
        *this_noise = noise_a + BitWidth(scalar_int64);
      }
    } else {
      *this_noise = noise_a + BitWidth(scalar_uint64);
    }
  }

  // Matrix multiplication operations.
  else if (IsMatMulCtPt(*node_def)) {
    uint64_t mul_noise = noise_a + BitWidth(error_params.B_plaintext());

    int32 reduce_dim_size = 0;
    if (!TryGetNodeAttr(*node_def, "reduce_dim_size", &reduce_dim_size) ||
        reduce_dim_size == -1) {
      std::cout << "WARNING: Could not determine dimension size to reduce in "
                   "MatMulCtPt. Noise budget may be under-provisioned."
                << std::endl;
      *this_noise = mul_noise;
    } else {
      *this_noise = mul_noise + BitWidth(reduce_dim_size);
    }
  } else if (IsMatMulPtCt(*node_def) || IsFastMatMulPtCt(*node_def)) {
    uint64_t mul_noise = noise_b + BitWidth(error_params.B_plaintext());

    std::string reduction;
    if (!TryGetNodeAttr(*node_def, "reduction", &reduction)) {
      std::cout << "WARNING: Could not determine reduction type for "
                   "MatMulPtCt. Noise budget may be under-provisioned."
                << std::endl;
      *this_noise = mul_noise;
    } else {
      if (reduction == "none") {
        *this_noise = mul_noise;
      } else if (reduction == "fast") {
        *this_noise =
            mul_noise +
            BitWidth(params.log_n);  // There are log_n ct-ct additions.
      } else if (reduction == "galois") {
        uint64_t rot_noise =
            BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
                kNumComponents, kLogGadgetBase, gadget_dimension));
        rot_noise = std::max(mul_noise, rot_noise) + 1;
        rot_noise += BitWidth(params.log_n);  // There are log_n rotations.
        rot_noise += BitWidth(params.log_n);  // There are log_n additions.
        *this_noise = rot_noise;
      } else {
        std::cout << "WARNING: Unknown reduction type for MatMulPtCt. Noise "
                     "budget may be under-provisioned."
                  << std::endl;
        *this_noise = mul_noise;
      }
    }
  }

  // Rotation operations.
  else if (IsRoll(*node_def)) {
    uint64_t rot_noise = BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
        kNumComponents, kLogGadgetBase, gadget_dimension));
    *this_noise = std::max(noise_b, rot_noise) + 1;
  } else if (IsReduceSumByRotation(*node_def)) {
    uint64_t rot_noise = BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
        kNumComponents, kLogGadgetBase, gadget_dimension));
    rot_noise = std::max(noise_b, rot_noise) + 1;
    rot_noise += BitWidth(params.log_n);  // There are log_n rotations.
    rot_noise += BitWidth(params.log_n);  // There are log_n ct-ct additions.
    *this_noise = rot_noise;
  } else if (IsFastReduceSumByRotation(*node_def)) {
    *this_noise =
        noise_a + BitWidth(params.log_n);  // There are log_n ct-ct additions.

    // Reduce sum (without rotation) operations.
  } else if (IsReduceSum(*node_def)) {
    int32 reduce_dim_size = 0;
    if (!TryGetNodeAttr(*node_def, "reduce_dim_size", &reduce_dim_size) ||
        reduce_dim_size == -1) {
      std::cout
          << "WARNING: Could not determine axis in reduce sum (ciphertext)."
          << std::endl;
      *this_noise = noise_a;
    } else {
      *this_noise = noise_a + BitWidth(reduce_dim_size);
    }
  }

  // Segment operations.
  else if (IsUnsortedCtSegmentSum(*node_def)) {
    uint64_t rot_noise = BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
        kNumComponents, kLogGadgetBase, gadget_dimension));
    rot_noise += BitWidth(params.log_n);  // There are at most log_n rotations.
    std::cout << "WARNING: ciphertext noise introduced by UnsortedCtSegmentSum "
                 "is data dependent and cannot be estimated. Use the noise "
                 "offset parameter to budget accordingly."
              << std::endl;
    *this_noise = std::max(noise_a, rot_noise) + 1 + 8;
  }

  // Conv2d operations. Convolution requires one multiplication and n - 1
  // additions, where n is the number of elements in the filter.
  else if (IsConv2d(*node_def)) {
    // Convolution performs one multiplication.
    if (IsCtPtConv2dOrTranspose(*node_def)) {
      *this_noise = noise_a + BitWidth(error_params.B_plaintext());
    } else if (IsPtCtConv2dOrTranspose(*node_def)) {
      *this_noise = BitWidth(error_params.B_plaintext()) + noise_b;
    } else if (IsCtCtConv2dOrTranspose(*node_def)) {
      *this_noise = noise_a + noise_b;
    }
    // Convolution performs n - 1 additions where n is the number of elements
    // in the filter.
    int32 filter_elems = 0;
    if (!TryGetNodeAttr(*node_def, "filter_num_elements", &filter_elems)) {
      std::cout << "WARNING: Could not determine filter num_elements when "
                   "estimating noise. Noise budget may be under-provisioned."
                << std::endl;
    } else {
      *this_noise += BitWidth(filter_elems - 1);
    }
  }

  else if (IsMaxUnpool2d(*node_def)) {
    // Max unpooling performs one multiplication.
    *this_noise = noise_a + BitWidth(error_params.B_plaintext());
    // Then n - 1 additions where n is the number of elements in the filter.
    std::vector<int32> pool_size;
    if (!TryGetNodeAttr(*node_def, "pool_size", &pool_size)) {
      std::cout << "WARNING: Could not determine pool size when "
                   "estimating noise. Noise budget may be under-provisioned."
                << std::endl;
    } else {
      int64_t total_pool_size = 1;
      for (auto const& size : pool_size) {
        total_pool_size *= size;
      }
      *this_noise += BitWidth(total_pool_size - 1);
    }
  }

  // hadal shape ops.
  else if (IsConcatCt(*node_def)) {
    // Fanins from 1 to n - 1 are the input tensors to be concatenated.
    // The first fanin is the axis. The noise is the maximum of the input
    // tensors.
    *this_noise = 0;
    for (int i = 1; i < node_view->NumRegularFanins(); ++i) {
      *this_noise = std::max(
          *this_noise, node_noise[node_view->GetRegularFanin(i).node_index()]);
    }
  }

  // Tensorflow Ops.
  else if (IsExpandDimsVariant(*node_def)) {
    *this_noise = node_noise[node_view->GetRegularFanin(0).node_index()];
  } else if (IsReshape(*node_def)) {
    *this_noise = node_noise[node_view->GetRegularFanin(0).node_index()];
  } else if (IsSplitV(*node_def)) {
    *this_noise = node_noise[node_view->GetRegularFanin(0).node_index()];
  }

  else if (IsDecrypt(*node_def)) {
    *this_noise = noise_b;
  }

  if constexpr (debug_noise_estimation) {
    std::cout << "\tNode " << node_def->name() << " noise bits: " << *this_noise
              << std::endl;
  }

  return OkStatus();
}

template <typename T>
Status EstimateNoiseGrowth(utils::MutableGraphView& graph_view,
                           utils::MutableNodeView const* autocontext,
                           ShellParams const& params,
                           uint64_t const noise_varaince, uint64_t* log_noise) {
  // Estimate the ciphertext noise growth by traversing the graph.
  int const num_nodes = graph_view.NumNodes();
  std::vector<uint64_t> node_noise(num_nodes);

  // Create RnsErrorParams.
  using ModularInt = rlwe::MontgomeryInt<T>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;
  std::vector<PrimeModulus*> main_moduli;
  for (auto const& q : params.qs) {
    using ModularIntParams = typename rlwe::MontgomeryInt<T>::Params;
    using NttParameters = rlwe::NttParameters<ModularInt>;

    RLWE_ASSIGN_OR_RETURN(std::unique_ptr<ModularIntParams const> mod_params_q,
                          ModularInt::Params::Create(q));
    RLWE_ASSIGN_OR_RETURN(NttParameters ntt_params_q,
                          rlwe::InitializeNttParameters<ModularInt>(
                              params.log_n, mod_params_q.get()));
    auto ntt_params_q_ptr =
        std::make_unique<NttParameters const>(std::move(ntt_params_q));
    auto modulus_q =
        new PrimeModulus{std::move(mod_params_q), std::move(ntt_params_q_ptr)};
    main_moduli.push_back(std::move(modulus_q));
  }
  auto error_params_or = rlwe::RnsErrorParams<ModularInt>::Create(
      params.log_n, main_moduli, {}, BitWidth(params.t), noise_varaince);
  if (!error_params_or.ok()) {
    return error_params_or.status();
  }
  rlwe::RnsErrorParams<ModularInt> error_params = error_params_or.value();

  uint64_t log_max_noise = 0;
  if constexpr (debug_noise_estimation) {
    std::cout << "Estimating noise growth." << std::endl;
  }
  for (int i = 0; i < num_nodes; ++i) {
    // Estimate the noise budget of this node.
    TF_RETURN_IF_ERROR(
        EstimateNodeNoise<T>(graph_view, i, node_noise, params, error_params));

    // If this is a decryption node, update the maximum node budget. Ensure the
    // decryption node uses the same autocontext node as the argument (for the
    // case when there are multiple).
    utils::MutableNodeView const* node_view = graph_view.GetNode(i);
    if (IsDecrypt(*node_view->node())) {
      TF_ASSIGN_OR_RETURN(bool is_same_autocontext,
                          DecryptUsesSameContext(node_view, autocontext));
      if (is_same_autocontext) {
        // If this is a decrypt node, and the context is the same, the noise
        // should never be zero. If it is, this means there is a problem with
        // the noise estimation.
        if (node_noise[i] == 0) {
          return errors::FailedPrecondition(
              "Noise budget of decrypt node is zero. Could not estimate noise "
              "growth. Check the noise estimation logic.");
        }

        // Update the maximum noise budget.
        log_max_noise = std::max(log_max_noise, node_noise[i]);
      }
    }
  }

  if constexpr (debug_noise_estimation) {
    std::cout << "Max noise bits: " << log_max_noise << std::endl;
  }

  if (log_max_noise == 0) {
    return errors::FailedPrecondition("Could not estimate noise growth.");
  }

  *log_noise = log_max_noise;
  return OkStatus();
}

Status ReplaceAutoparamWithContext(utils::MutableGraphView& graph_view,
                                   utils::MutableNodeView* autocontext,
                                   ShellParams const& params,
                                   ShellAutoParams const& auto_params) {
  int autocontext_node_index = autocontext->node_index();

  if constexpr (debug_graph) {
    std::cout << "Removing AutoShellContext node: "
              << autocontext->node()->DebugString() << std::endl;
  }

  // Create the new inputs for the ShellContext node.
  std::string log_n_name = autocontext->GetName() + "/log_n";
  std::string qs_name = autocontext->GetName() + "/main_moduli";
  std::string ps_name = autocontext->GetName() + "/aux_moduli";
  std::string t_name = autocontext->GetName() + "/plaintext_modulus";
  std::string noise_var_name = autocontext->GetName() + "/noise_variance";
  std::string seed_str_name = autocontext->GetName() + "/seed";

  utils::Mutation* mutation = graph_view.GetMutationBuilder();
  std::string device = autocontext->GetDevice();
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<uint64_t>(params.log_n, mutation, log_n_name, device));
  TF_RETURN_IF_ERROR(AddScalarConstNode<std::vector<uint64_t>>(
      params.qs, mutation, qs_name, device));
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<std::vector<uint64_t>>({}, mutation, ps_name, device));
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<uint64_t>(params.t, mutation, t_name, device));
  TF_RETURN_IF_ERROR(AddScalarConstNode<uint64_t>(
      auto_params.noise_variance, mutation, noise_var_name, device));
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<std::string>("", mutation, seed_str_name, device));

  // Replace the AutoShellContext node with ShellContextImport64.
  NodeDef shell_context_import_node;
  shell_context_import_node.set_op(kShellContext);
  std::string new_name = autocontext->GetName();
  new_name = new_name.insert(new_name.find_last_of('/') + 1, "Optimized");
  shell_context_import_node.set_name(new_name);
  shell_context_import_node.set_device(device);
  shell_context_import_node.add_input(log_n_name);
  shell_context_import_node.add_input(qs_name);
  shell_context_import_node.add_input(ps_name);
  shell_context_import_node.add_input(t_name);
  shell_context_import_node.add_input(noise_var_name);
  shell_context_import_node.add_input(seed_str_name);
  if constexpr (debug_graph) {
    std::cout << "Adding new ShellContext node: "
              << shell_context_import_node.DebugString() << std::endl;
  }
  Status status;
  mutation->AddNode(std::move(shell_context_import_node), &status);
  TF_RETURN_IF_ERROR(status);

  // Update all fanouts of the AutoShellContext node to point to the new
  // ShellContext node.
  auto const& all_fanouts = autocontext->GetRegularFanouts();
  for (int i = 0; i < static_cast<int>(all_fanouts.size()); ++i) {
    auto const& fanouts_by_port = all_fanouts[i];
    for (auto const& fanout : fanouts_by_port) {
      mutation->AddOrUpdateRegularFanin(fanout.node_view(), fanout.index(),
                                        {new_name, i});
      if constexpr (debug_graph) {
        std::cout << "Updating fanout: " << fanout.node_view()->node()->name()
                  << " index: " << fanout.index()
                  << " to ShellContext output port: " << i << std::endl;
      }
    }
  }

  mutation->RemoveNode(graph_view.GetNode(autocontext_node_index));
  for (auto const& fanin : autocontext->GetRegularFanins()) {
    // When there are multiple autocontexts, the fanins may be shared. Only
    // remove the fanin if it is not shared.
    if (fanin.node_view()->NumRegularFanouts() == 1) {
      mutation->RemoveNode(fanin.node_view());
    }
  }

  // This op does not currently handle controlling or controlled fanins/outs.
  if (autocontext->GetControllingFanins().size() > 0 ||
      autocontext->GetControlledFanouts().size() > 0) {
    return errors::Unimplemented(
        "Replacing AutoShellContext with ShellContext does not currently "
        "support controlling or controlled fanins/outs.");
  }

  TF_RETURN_IF_ERROR(mutation->Apply());

  return OkStatus();
}

Status OptimizeAutocontext(utils::MutableGraphView& graph_view,
                           utils::MutableNodeView* autocontext) {
  // Use GetScalarConstValue to get value of plaintext modulus,
  // etc.
  ShellAutoParams auto_params;
  TF_RETURN_IF_ERROR(GetAutoShellContextParams(autocontext, auto_params));

  // Find the maximum scaling factor used in the graph and use this to set the
  // plaintext modulus t. The maximum plaintext bits * the maximum scaling
  // factor is the maximum size. This is too conservative an estimate, so
  // for now, it is disabled.
  // TF_ASSIGN_OR_RETURN(int64_t max_sf,
  //                     MaxScalingFactor(graph_view, autocontext));
  // uint64_t sf_bits = BitWidth(max_sf);
  uint64_t total_plaintext_bits = auto_params.cleartext_bits;

  if constexpr (debug_moduli) {
    // std::cout << "Max bits of scaling factor upon decryption: " << sf_bits
    //           << std::endl;
    std::cout << "Total Cleartext Bits: " << total_plaintext_bits << std::endl;
  }
  if (total_plaintext_bits >= kMaxPrimeBitsCiphertext) {
    std::cout << "ERROR: Total plaintext size (" << total_plaintext_bits
              << ") >= " << kMaxPrimeBitsCiphertext
              << ". This is not supported." << std::endl;
    return errors::FailedPrecondition("Total plaintext size exceeds ",
                                      kMaxPrimeBitsCiphertext,
                                      ". This is not supported.");
  }

  // Do binary search to find the optimal size of the ciphertext modulus q.
  // Choose a window of moduli to search starting slightly larger than the
  // plaintext modulus. Store the best moduli in qs and ring degree in log_n.
  uint64_t log_q_l = total_plaintext_bits + 2;
  uint64_t log_q_r = total_plaintext_bits + 210;
  uint64_t log_q = (log_q_r + log_q_l) / 2;
  uint64_t log_max_noise = 0;
  int64_t total_ct_bits = 0;
  ShellParams params;

  while (log_q_l <= log_q_r && log_q < log_q_r + 10) {
    // The + 10 above is for the case when the binary search ends on parameters
    // which are not suitable for shell. In this case, the search will continue
    // 10 log_q's higher than the last suitable parameters found.
    Status params_status =
        ChooseShellParams(params, total_plaintext_bits, log_q);
    if (!params_status.ok()) {
      // No suitable parameters found. Try increasing log_q.
      if constexpr (debug_moduli) {
        std::cout << "No suitable parameters found for log_q: " << log_q
                  << ". Trying larger log_q." << std::endl;
      }
      log_q += 1;
      continue;
    }

    TF_RETURN_IF_ERROR(EstimateNoiseGrowth<uint64_t>(
        graph_view, autocontext, params, auto_params.noise_variance,
        &log_max_noise));

    total_ct_bits =
        BitWidth(params.t) + log_max_noise + auto_params.noise_offset_bits;

    // Adjust the noise budget to account for the encryption noise.
    if (total_ct_bits > static_cast<uint64_t>(log_q)) {
      if constexpr (debug_moduli) {
        std::cout << "Noise budget exceeded "
                  << "(plaintext bits: " << BitWidth(params.t)
                  << " + max noise bits: " << log_max_noise
                  << " + noise offset: " << auto_params.noise_offset_bits
                  << " = " << total_ct_bits << " > Q bits: " << log_q << ")."
                  << std::endl;
      }
      log_q_l = log_q + 1;
    } else {
      if constexpr (debug_moduli) {
        std::cout << "Noise budget over provisioned "
                  << "(plaintext bits: " << BitWidth(params.t)
                  << " + max noise bits: " << log_max_noise
                  << " + noise offset: " << auto_params.noise_offset_bits
                  << " = " << total_ct_bits << " < Q bits: " << log_q << ")."
                  << std::endl;
      }
      log_q_r = std::min(log_q_r - 1, log_q - 1);
    }
    log_q = (log_q_r + log_q_l) / 2;
    if constexpr (debug_moduli) {
      std::cout << "New log_q: " << log_q << std::endl;
    }
  }

  if (log_q >= total_plaintext_bits + 210) {
    std::cout
        << "ERROR: Could not find suitable parameters. RNS chain requires "
        << total_ct_bits << " bits (maximum search space "
        << total_plaintext_bits + 210 << " bits)." << std::endl;
    return errors::FailedPrecondition("Could not find suitable parameters.");
  }

  TF_RETURN_IF_ERROR(ChooseShellParams(params, total_plaintext_bits, log_q));
  if constexpr (debug_output_params) {
    std::cout << "Selected BGV parameters:" << std::endl;
    std::cout << "log_n: " << params.log_n << std::endl;
    std::cout << "t: " << params.t << " (" << BitWidth(params.t)
              << " bits, min:" << total_plaintext_bits << ")" << std::endl;
    std::cout << "qs: ";
    uint64_t total_ct_bits = 0;
    for (auto const& q : params.qs) {
      std::cout << q << " ";
      total_ct_bits += BitWidth(q);
    }
    std::cout << " (" << total_ct_bits << " bits, min:"
              << BitWidth(params.t) + log_max_noise +
                     auto_params.noise_offset_bits
              << " = t:" << BitWidth(params.t) << " + noise:" << log_max_noise
              << " + offset:" << auto_params.noise_offset_bits << ")"
              << std::endl;
  }

  TF_RETURN_IF_ERROR(ReplaceAutoparamWithContext(graph_view, autocontext,
                                                 params, auto_params));
  return OkStatus();
}

}  // namespace

ModuliAutotuneOptimizer::ModuliAutotuneOptimizer() {}

Status ModuliAutotuneOptimizer::Init(
    tensorflow::RewriterConfig_CustomGraphOptimizer const* config) {
  return OkStatus();
}

Status ModuliAutotuneOptimizer::Optimize(Cluster* cluster,
                                         GrapplerItem const& item,
                                         GraphDef* optimized_graph) {
  GrapplerItem mutable_item(item);
  Status status;
  utils::MutableGraphView graph_view(&mutable_item.graph, &status);
  TF_RETURN_IF_ERROR(status);

  // Topological sort so all subsequent traversals are in order.
  TF_RETURN_IF_ERROR(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  // Optimize each autocontext op in the graph.
  utils::MutableNodeView* autocontext = GetNextAutoShellContextNode(graph_view);
  while (autocontext != nullptr) {
    TF_RETURN_IF_ERROR(OptimizeAutocontext(graph_view, autocontext));
    autocontext = GetNextAutoShellContextNode(graph_view);
  }

  if constexpr (debug_graph) {
    std::cout << "Optimized graph: " << std::endl;
    int const num_nodes = graph_view.NumNodes();
    for (int i = 0; i < num_nodes; ++i) {
      std::cout << graph_view.GetNode(i)->node()->DebugString() << std::endl;
    }
  }

  *optimized_graph = std::move(mutable_item.graph);

  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER(ModuliAutotuneOptimizer);

}  // namespace grappler
}  // namespace tensorflow
