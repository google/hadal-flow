load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package")

filegroup(
    name = "hadal_ml_files",
    srcs = glob(
        [
            "*.py",
        ],
        exclude = [
            "test",
        ],
    ),
)

py_library(
    name = "hadal_ml",
    srcs = [
        ":hadal_ml_files",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
    ],
)

py_package(
    name = "hadal_ml_pkg",
    visibility = ["//:__pkg__"],
    deps = [
        ":hadal_ml",
    ],
)
