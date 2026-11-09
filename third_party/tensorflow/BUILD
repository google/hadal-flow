load("@pip//:requirements.bzl", "requirement")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary")
load("//tools:dynamic_genrule.bzl", "dynamic_genrule")

py_binary(
    name = "extract_tf_includes",
    srcs = [":extract_tf_includes.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tensorflow"),
    ],
)

py_binary(
    name = "extract_tf_lib",
    srcs = [":extract_tf_lib.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tensorflow"),
    ],
)

dynamic_genrule(
    name = "hermetic_tf_headers",
    tool = ":extract_tf_includes",
)

genrule(
    name = "hermetic_tf_lib",
    outs = ["libtensorflow_framework.so"],
    cmd = "$(execpath :extract_tf_lib) $@",
    tools = [":extract_tf_lib"],
)

cc_library(
    name = "tf_headers",
    hdrs = [":hermetic_tf_headers"],
    includes = ["hermetic_tf_headers/include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_lib",
    data = [":hermetic_tf_lib"],
    visibility = ["//visibility:public"],
    deps = [":hermetic_tf_lib"],
)

# At runtime, libtensorflow_framework.so will be provided by pip, so this rule
# uses system_provided = 1 to indicate that the library is only needed to build
# but can be skipped when packaging.
cc_import(
    name = "hermetic_tf",
    hdrs = [":tf_headers"],
    interface_library = "libtensorflow_framework.so",
    system_provided = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":tf_headers",
        ":tf_lib",
    ],
)
