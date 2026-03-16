#include "ct_pt.h"

#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr bool const debug = false;

char const* GetOpFromCtPt(NodeDef const& node, bool is_ct_pt) {
  if (IsAddCtPt(node)) {
    return is_ct_pt ? kAddCtPt : kAddPtPt;
  } else if (IsSubCtPt(node)) {
    return is_ct_pt ? kSubCtPt : kAddPtPt;  // PtPt replaces sub with add.
  } else if (IsMulCtPt(node)) {
    return is_ct_pt ? kMulCtPt : kMulPtPt;
  }

  return nullptr;
}

struct ReorderArith {
  int shell_context_node_index;
  int outer_node_index;
  int inner_node_index;
  int outer_pt_node_index;
  int inner_ct_node_index;
  int inner_pt_node_index;
};

void PrintReorderArith(utils::MutableGraphView& graph_view,
                       ReorderArith const& reorder) {
  auto const* outer_node = graph_view.GetNode(reorder.outer_node_index)->node();
  auto const* inner_node = graph_view.GetNode(reorder.inner_node_index)->node();
  auto const* outer_pt_node =
      graph_view.GetNode(reorder.outer_pt_node_index)->node();
  auto const* inner_ct_node =
      graph_view.GetNode(reorder.inner_ct_node_index)->node();
  auto const* inner_pt_node =
      graph_view.GetNode(reorder.inner_pt_node_index)->node();

  std::cout << outer_node->name() << " ( " << inner_node->name() << " ( "
            << inner_ct_node->name() << " , " << inner_pt_node->name() << " ), "
            << outer_pt_node->name() << " ) " << std::endl;
}

// Returns true if the node_index points to the outermost op of the pattern
// outer_op(inner_op(ct, pt), pt) and fills the ReorderArith struct accordingly.
// If the outer_op is add or sub, the inner_op must be add or sub.
// If instead the outer_op is mul, the inner_op must be mul.
// If the inner op is used elsewhere (has fanout>1), the pattern is not matched.
bool FindAddOrSub(utils::MutableGraphView& graph_view, int node_index,
                  ReorderArith* reorder) {
  // Check given node is op(ct, pt).
  auto const* outer_node_view = graph_view.GetNode(node_index);
  auto const* outer_node_def = outer_node_view->node();

  if (!IsAddCtPt(*outer_node_def) && !IsSubCtPt(*outer_node_def) &&
      !IsMulCtPt(*outer_node_def)) {
    return false;
  }

  // Next, check the feed node ct at input 0 is the output of another
  // CtPt op.
  auto const& outer_fanin_0 = outer_node_view->GetRegularFanin(0);
  auto const* context_node_view = outer_fanin_0.node_view();

  // The first input of the outer node is the inner op.
  auto const& outer_fanin_1 = outer_node_view->GetRegularFanin(1);
  auto const* inner_node_view = outer_fanin_1.node_view();
  auto const* inner_node_def = inner_node_view->node();

  // First check the inner node is not used elsewhere in the graph, in which
  // case it must be computed and cannot be optimized away.
  if (inner_node_view->NumRegularFanouts() != 1) {
    return false;
  }

  auto const& outer_fanin_2 = outer_node_view->GetRegularFanin(2);
  auto const* outer_pt_node_view = outer_fanin_2.node_view();

  // If the outer op is add or sub, the inner op must be add or sub as well.
  if (!IsMulCtPt(*outer_node_def) && !IsAddCtPt(*inner_node_def) &&
      !IsSubCtPt(*inner_node_def)) {
    return false;
  }

  // If the outer op is mul, the inner op must be mul as well.
  if (IsMulCtPt(*outer_node_def) && !IsMulCtPt(*inner_node_def)) {
    return false;
  }

  auto const& inner_fanin_0 = inner_node_view->GetRegularFanin(0);
  auto const* inner_context_node_view = inner_fanin_0.node_view();

  // If the contexts do not match, the pattern should not be matched.
  if (context_node_view->node_index() != inner_context_node_view->node_index())
    return false;

  // If the devices do not match, the pattern should not be matched.
  if (outer_node_def->device() != inner_node_view->node()->device())
    return false;

  auto const& inner_fanin_1 = inner_node_view->GetRegularFanin(1);
  auto const* inner_ct_node_view = inner_fanin_1.node_view();

  auto const& inner_fanin_2 = inner_node_view->GetRegularFanin(2);
  auto const* inner_pt_node_view = inner_fanin_2.node_view();

  ReorderArith new_reorder{
      .shell_context_node_index = context_node_view->node_index(),
      .outer_node_index = node_index,
      .inner_node_index = inner_node_view->node_index(),
      .outer_pt_node_index = outer_pt_node_view->node_index(),
      .inner_ct_node_index = inner_ct_node_view->node_index(),
      .inner_pt_node_index = inner_pt_node_view->node_index()};

  if constexpr (debug) {
    std::cout << "Found pattern:";
    PrintReorderArith(graph_view, new_reorder);
  }

  *reorder = new_reorder;

  return true;
}

// This function replaces the pattern outer_op(inner_op(ct, pt), pt) with
// outer_op(ct, inner_op(pt, pt)).
Status ApplyReorderArith(utils::MutableGraphView& graph_view,
                         ReorderArith const& reorder,
                         std::vector<bool>* nodes_to_delete) {
  GraphDef const* graph = graph_view.graph();
  utils::Mutation* mutation = graph_view.GetMutationBuilder();
  Status status;

  // First replace the inner node with a pt pt node.
  NodeDef const& inner_node_def = graph->node(reorder.inner_node_index);
  NodeDef new_pt_pt_inner;

  auto const new_inner_op = GetOpFromCtPt(inner_node_def, /*is_ct_pt=*/false);
  if (new_inner_op == nullptr) {
    return errors::Internal("Inner node is not supported type.");
  }
  new_pt_pt_inner.set_op(new_inner_op);
  // Name of the new node needs to be the same as the old, even though the
  // op is different, so downstream nodes can still find it.
  new_pt_pt_inner.set_name(inner_node_def.name());
  new_pt_pt_inner.set_device(inner_node_def.device());

  NodeDef const& shell_context_node =
      graph->node(reorder.shell_context_node_index);
  new_pt_pt_inner.add_input(shell_context_node.name());
  NodeDef const& inner_pt = graph->node(reorder.inner_pt_node_index);
  new_pt_pt_inner.add_input(inner_pt.name());
  NodeDef const& outer_pt = graph->node(reorder.outer_pt_node_index);
  new_pt_pt_inner.add_input(outer_pt.name());

  // Replace the outer node with a ct pt op, where pt comes from
  // new_pt_pt_inner created above.
  NodeDef const& outer_node_def = graph->node(reorder.outer_node_index);
  NodeDef new_outer;

  auto const new_outer_op = GetOpFromCtPt(outer_node_def, /*is_ct_pt=*/true);
  if (new_outer_op == nullptr) {
    return errors::Internal("Inner node is not supported type.");
  }
  new_outer.set_op(new_outer_op);
  // Name of the new node needs to be the same as the old, even though the
  // op is different, so downstream nodes can still find it.
  new_outer.set_name(outer_node_def.name());
  new_outer.set_device(outer_node_def.device());

  NodeDef const& inner_ct = graph->node(reorder.inner_ct_node_index);
  new_outer.add_input(shell_context_node.name());
  new_outer.add_input(inner_ct.name());
  new_outer.add_input(new_pt_pt_inner.name());

  if constexpr (debug) {
    std::cout << "New outer node: " << new_outer.DebugString() << std::endl;
    std::cout << "New inner node: " << new_pt_pt_inner.DebugString()
              << std::endl;
  }

  // Add the new nodes to the graph.
  mutation->AddNode(std::move(new_outer), &status);
  TF_RETURN_IF_ERROR(status);
  mutation->AddNode(std::move(new_pt_pt_inner), &status);
  TF_RETURN_IF_ERROR(status);

  (*nodes_to_delete)[reorder.outer_node_index] = true;
  (*nodes_to_delete)[reorder.inner_node_index] = true;

  return OkStatus();
}

}  // namespace

CtPtOptimizer::CtPtOptimizer() {}

Status CtPtOptimizer::Init(
    tensorflow::RewriterConfig_CustomGraphOptimizer const* config) {
  return OkStatus();
}

Status CtPtOptimizer::Optimize(Cluster* cluster, GrapplerItem const& item,
                               GraphDef* optimized_graph) {
  GrapplerItem mutable_item(item);
  Status status;
  utils::MutableGraphView graph_view(&mutable_item.graph, &status);
  TF_RETURN_IF_ERROR(status);

  // Topological sort and process the nodes in reverse.
  TF_RETURN_IF_ERROR(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  bool finished = false;
  while (!finished) {
    int const num_nodes = mutable_item.graph.node_size();
    std::vector<bool> nodes_to_delete(num_nodes);
    finished = true;

    for (int i = num_nodes - 1; i >= 0; --i) {
      if (nodes_to_delete[i]) {
        continue;
      }

      // Remap op( op(ct, pt), pt) to op(ct, op(pt, pt)).
      ReorderArith reorder;
      if (FindAddOrSub(graph_view, i, &reorder)) {
        TF_RETURN_IF_ERROR(
            ApplyReorderArith(graph_view, reorder, &nodes_to_delete));
        finished = false;
      }
    }

    // Remove nodes.
    utils::Mutation* mutation = graph_view.GetMutationBuilder();
    for (int i = 0; i < num_nodes; ++i) {
      if (nodes_to_delete[i]) {
        mutation->RemoveNode(graph_view.GetNode(i));
      }
    }
    TF_RETURN_IF_ERROR(mutation->Apply());
  }

  *optimized_graph = std::move(mutable_item.graph);

  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER(CtPtOptimizer);

}  // namespace grappler
}  // namespace tensorflow