#include "filesystem_distribution.h"

#include <algorithm>
#include <cstdlib>
#include <map>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "utils.h"

namespace tensorflow {
namespace grappler {

FilesystemDistributionOptimizer::FilesystemDistributionOptimizer() {}

Status FilesystemDistributionOptimizer::Init(
    tensorflow::RewriterConfig_CustomGraphOptimizer const* config) {
  return OkStatus();
}

Status FilesystemDistributionOptimizer::Optimize(Cluster* cluster,
                                                 GrapplerItem const& item,
                                                 GraphDef* optimized_graph) {
  char const* env_path = std::getenv("HADAL_FILESYSTEM_PATH");
  if (!env_path) {
    *optimized_graph = item.graph;
    return OkStatus();
  }
  string base_path = string(env_path);
  if (!base_path.empty() && base_path.back() != '/') {
    base_path += "/";
  }

  GrapplerItem mutable_item(item);
  Status status;
  GraphProperties properties(mutable_item);
  status = properties.InferStatically(true);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to infer shapes: " << status.message();
  }
  utils::MutableGraphView graph_view(&mutable_item.graph, &status);
  TF_RETURN_IF_ERROR(status);

  int const num_nodes = mutable_item.graph.node_size();
  utils::Mutation* mutation = graph_view.GetMutationBuilder();

  std::map<int, NodeDef> nodes_to_replace;
  std::vector<bool> nodes_to_delete(num_nodes, false);

  int const_id = 0;

  for (int i = 0; i < num_nodes; ++i) {
    NodeDef const* original_dst_node = graph_view.GetNode(i)->node();
    string dst_device = original_dst_node->device();

    NodeDef* current_dst_node_ptr = nullptr;
    if (nodes_to_replace.count(i)) {
      current_dst_node_ptr = &nodes_to_replace[i];
    }

    int num_inputs = original_dst_node->input_size();
    bool node_modified = false;
    NodeDef new_dst_node;
    if (current_dst_node_ptr) {
      new_dst_node = *current_dst_node_ptr;
    } else {
      new_dst_node = *original_dst_node;
    }

    for (int j = 0; j < num_inputs; ++j) {
      string input_name_full = new_dst_node.input(j);
      if (input_name_full.rfind("^", 0) == 0) continue;

      string src_node_name = input_name_full;
      int output_index = 0;
      size_t colon_pos = input_name_full.find(':');
      if (colon_pos != string::npos) {
        src_node_name = input_name_full.substr(0, colon_pos);
        string index_str = input_name_full.substr(colon_pos + 1);
        if (!index_str.empty() &&
            std::all_of(index_str.begin(), index_str.end(), ::isdigit)) {
          output_index = std::stoi(index_str);
        }
      }

      auto const* src_node_view = graph_view.GetNode(src_node_name);
      if (!src_node_view) continue;
      NodeDef const* src_node = src_node_view->node();

      string src_device = src_node->device();

      if (src_device.empty() || dst_device.empty()) continue;
      if (src_device == dst_device) continue;

      if (IsKeyGen(*src_node)) {
        LOG(WARNING)
            << "WARNING: Secret key is being transfered between tf.deivces";
      }

      if (!(NodeOutputsCtOrPt(*src_node) || IsRotationKeyGen(*src_node) ||
            IsFastRotationKeyGen(*src_node)))
        continue;

      string unique_suffix = "_" + std::to_string(i) + "_" + std::to_string(j) +
                             "_" + std::to_string(const_id++);

      string safe_src = src_node->name();
      std::replace(safe_src.begin(), safe_src.end(), '/', '_');
      string safe_dst = new_dst_node.name();
      std::replace(safe_dst.begin(), safe_dst.end(), '/', '_');
      string filename =
          base_path + safe_src + "_to_" + safe_dst + unique_suffix + ".bin";

      LOG(INFO) << "Transfering shell tensor via file " << filename;

      // 1. Path Const (Src Device)
      string path_const_name = src_node->name() + "_path" + unique_suffix;
      NodeDef path_const;
      path_const.set_name(path_const_name);
      path_const.set_op("Const");
      path_const.set_device(src_device);

      AttrValue dtype_attr;
      dtype_attr.set_type(DT_STRING);
      path_const.mutable_attr()->insert({"dtype", dtype_attr});

      AttrValue value_attr;
      TensorProto* t = value_attr.mutable_tensor();
      t->set_dtype(DT_STRING);
      t->add_string_val(filename);
      t->mutable_tensor_shape();  // Scalar
      path_const.mutable_attr()->insert({"value", value_attr});

      mutation->AddNode(std::move(path_const), &status);
      TF_RETURN_IF_ERROR(status);

      // 2. Save (Src Device)
      string save_name = src_node->name() + "_save" + unique_suffix;
      NodeDef save_node;
      save_node.set_name(save_name);
      save_node.set_op("SaveShellTensor");
      save_node.set_device(src_device);
      save_node.add_input(input_name_full);
      save_node.add_input(path_const_name);

      AttrValue t_variant;
      t_variant.set_type(DT_VARIANT);
      save_node.mutable_attr()->insert({"T", t_variant});

      mutation->AddNode(std::move(save_node), &status);
      TF_RETURN_IF_ERROR(status);

      // 3. Load (Dst Device)
      string load_name = new_dst_node.name() + "_load" + unique_suffix;
      NodeDef load_node;
      load_node.set_name(load_name);
      load_node.set_op("LoadShellTensor");
      load_node.set_device(dst_device);
      load_node.add_input(save_name);  // Path from Save output

      load_node.mutable_attr()->insert({"T", t_variant});

      auto const& output_props = properties.GetOutputProperties(src_node_name);
      if (output_index >= 0 &&
          output_index < static_cast<int>(output_props.size())) {
        TensorShapeProto shape = output_props[output_index].shape();
        for (int k = 0; k < shape.dim_size(); ++k) {
          if (shape.dim(k).size() < -1) {
            LOG(WARNING) << "GraphProperties returned invalid dimension size "
                         << shape.dim(k).size() << " for " << src_node_name
                         << ":" << output_index << ". Sanitizing to -1.";
            shape.mutable_dim(k)->set_size(-1);
          }
        }

        AttrValue shape_attr;
        *shape_attr.mutable_shape() = shape;
        load_node.mutable_attr()->insert({"output_shape", shape_attr});
      }

      mutation->AddNode(std::move(load_node), &status);
      TF_RETURN_IF_ERROR(status);

      // Update input
      new_dst_node.set_input(j, load_name);
      node_modified = true;
    }

    if (node_modified) {
      nodes_to_replace[i] = new_dst_node;
      nodes_to_delete[i] = true;
    }
  }

  // Apply replacements
  for (auto const& [idx, new_node] : nodes_to_replace) {
    NodeDef node_copy = new_node;
    mutation->AddNode(std::move(node_copy), &status);
    TF_RETURN_IF_ERROR(status);
  }

  // Remove old nodes
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(graph_view.GetNode(i));
    }
  }

  TF_RETURN_IF_ERROR(mutation->Apply());
  *optimized_graph = std::move(mutable_item.graph);

  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER(FilesystemDistributionOptimizer);

}  // namespace grappler
}  // namespace tensorflow
