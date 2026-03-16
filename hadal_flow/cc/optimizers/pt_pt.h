#pragma once

#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/functions.h"

namespace tensorflow {
namespace grappler {

class PtPtOptimizer : public CustomGraphOptimizer {
 public:
  PtPtOptimizer();

  Status Init(
      tensorflow::RewriterConfig_CustomGraphOptimizer const* config) override;

  string name() const override { return name_; }

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, GrapplerItem const& item,
                  GraphDef* optimized_graph) override;

 private:
  string const name_ = "PtPtOptimizer";
};

}  // namespace grappler
}  // namespace tensorflow