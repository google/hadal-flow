# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

shell_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_shell_ops.so")
)

# Based on https://github.com/openvinotoolkit/openvino_tensorflow/blob/d9dcb9d4c5932d0a8e9a3633d4134ae5841af6c1/python/openvino_tensorflow/__init__.in.py
# Anther approach using higher level APIs can be found here:
# https://stackoverflow.com/questions/74219568/optimize-and-resave-saved-model-with-grappler

from tensorflow.python.framework import ops
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.training import saver
from tensorflow.python.util import nest
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.core.function.polymorphism import function_type as function_type_lib

all_shell_optimizers = [
    "CtPtOptimizer",
    "PtPtOptimizer",
    "ModuliAutotuneOptimizer",
    "FilesystemDistributionOptimizer",
]


def optimize_shell_graph(
    func, optimizers=all_shell_optimizers, skip_convert_to_constants=False
):
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    for optimizer in optimizers:
        custom_optimizer = rewriter_config.custom_optimizers.add()
        custom_optimizer.name = optimizer

    # Converting var2consts for larger models might take a long time
    if not skip_convert_to_constants:
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(
            func, lower_control_flow=False, aggressive_inlining=True
        )
    else:
        frozen_func = func

    meta_graph_def = saver.export_meta_graph(
        graph_def=frozen_func.graph.as_graph_def(add_shapes=False),
        graph=frozen_func.graph,
    )

    # print("orig graph def", meta_graph_def)

    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in frozen_func.outputs:
        fetch_collection.node_list.value.append(array.name)

    # Grappler determines fetch ops from collection 'train_op'.
    meta_graph_def.collection_def[ops.GraphKeys.TRAIN_OP].CopyFrom(fetch_collection)

    # For a clean slate, create a new grappler session config as below
    #   # grappler_session_config = config_pb2.ConfigProto()
    # But to retain other settings, like soft device placement, copy from the
    # existing config.
    grappler_session_config = context.context().config

    grappler_session_config.graph_options.rewrite_options.CopyFrom(rewriter_config)
    optimized_graph_def = tf_optimizer.OptimizeGraph(
        grappler_session_config, meta_graph_def, graph_id=b"tf_graph"
    )

    # print("opt graph def", optimized_graph_def)

    # Swap original function with optimized function in TF's context
    for f in optimized_graph_def.library.function:
        while context.context().has_function(f.signature.name):
            context.context().remove_function(f.signature.name)

    try:
        optimized_func = wrap_function.function_from_graph_def(
            optimized_graph_def,
            [tensor.name for tensor in frozen_func.inputs],
            [tensor.name for tensor in frozen_func.outputs],
        )
    except Exception as e:
        raise ValueError(
            "Could not wrap the optimized graph. Did the shell optimizer remove"
            " some of the inputs or outputs? Original error: " + str(e)
        )

    optimized_func.graph.structured_outputs = nest.pack_sequence_as(
        func.graph.structured_outputs,
        optimized_func.graph.structured_outputs,
        expand_composites=True,  # required for extension types e.g. ShellTensor
    )

    optimized_func.graph.structured_input_signature = func.structured_input_signature

    # `optimized_func` is a WrappedFunction holding an AtomicFunction which
    # derives a `function_type` from the structured_input_signature and
    # structured_outputs. This is used to flatten the arguments when calling the
    # function, which usually isn't important when the arguments are just
    # Tensors but when they are composite (e.g. ShellTensor) the flattening
    # becomes important.
    #
    # There are two bugs in Tensorflow that make this tricky.
    # 1) First, we need to force update the `function_type` to reflect the new
    # structured input and output signatures. Ideally, when
    # function_from_graph_def() calls prune to create the new WrappedFunction,
    # it would correctly set the pruned_graph.structured_input_signature instead
    # of None, and it would set the structured_outputs to the "structured"
    # typespec of the outputs, instead of the flattened version. Once
    # b/129646028 is fixed (prune supports composite tensors), this can
    # hopefully be removed. For more info, see line 390 of
    # tensorflow/python/eager/wrap_function.py in tensorflow 2.16.1.
    # 2) Second, after calling the optimized_func, the output args are never
    # unflattened. This is a bug in TensorFlow and a fix PR is submitted at
    # https://github.com/tensorflow/tensorflow/pull/67612.
    # For now, we require calling code to run something like:
    # my_func_output = optimized_func.function_type.pack_output(my_func_output)

    updated_fn_type = function_type_lib.from_structured_signature(
        optimized_func.graph.structured_input_signature,
        optimized_func.graph.structured_outputs,
        optimized_func.graph.function_captures.capture_types,
    )
    optimized_func._function_type = updated_fn_type

    return optimized_func


# Here is a method to enable custom optimizers described by
# https://github.com/tensorflow/tensorflow/issues/55451#issuecomment-1147065792
def enable_optimization(optimizers=all_shell_optimizers):
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    for optimizer in optimizers:
        custom_optimizer = rewriter_config.custom_optimizers.add()
        custom_optimizer.name = optimizer
    grappler_session_config = context.context().config
    grappler_session_config.graph_options.rewrite_options.CopyFrom(rewriter_config)

    grappler_options = context.FunctionCallOptions(config_proto=grappler_session_config)
    context.context().function_call_options = grappler_options
