#pragma once

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/utils/graph_view.h"

namespace tensorflow {
namespace grappler {

class FilesystemDistributionOptimizer : public CustomGraphOptimizer {
 public:
  FilesystemDistributionOptimizer();
  ~FilesystemDistributionOptimizer() override = default;

  Status Init(
      tensorflow::RewriterConfig_CustomGraphOptimizer const* config) override;
  Status Optimize(Cluster* cluster, GrapplerItem const& item,
                  GraphDef* optimized_graph) override;

  string name() const override { return "FilesystemDistributionOptimizer"; }
  bool UsesFunctionLibrary() const override { return false; }
};

}  // namespace grappler
}  // namespace tensorflow
