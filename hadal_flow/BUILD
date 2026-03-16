load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package")

filegroup(
    name = "shell_ops_src",
    srcs = glob([
        "cc/kernels/*.cc",
        "cc/kernels/*.h",
    ]),
)

filegroup(
    name = "shell_optimizer_src",
    srcs = glob([
        "cc/optimizers/*.cc",
        "cc/optimizers/*.h",
    ]),
)

cc_binary(
    name = "python/_shell_ops.so",
    srcs = [
        ":shell_ops_src",
        ":shell_optimizer_src",
    ],
    copts = [
        "-pthread",
        "-fPIC",
        "-fpermissive",  # For EMP's usage of EVP_sha3_256 in openssl
        "-maes",  # EMP
        "-mssse3",  # EMP
    ],
    linkshared = 1,
    deps = [
        "//third_party/emp:emp-sh2pc",
        "//third_party/tensorflow:hermetic_tf",
        "@shell_encryption//shell_encryption:context",
        "@shell_encryption//shell_encryption:montgomery",
        "@shell_encryption//shell_encryption:polynomial",
        "@shell_encryption//shell_encryption:transcription",
        "@shell_encryption//shell_encryption/prng:hkdf_prng",
        "@shell_encryption//shell_encryption/rns:finite_field_encoder",
        "@shell_encryption//shell_encryption/rns:rns_ciphertext",
        "@shell_encryption//shell_encryption/rns:rns_context",
        "@shell_encryption//shell_encryption/rns:rns_error_params",
        "@shell_encryption//shell_encryption/rns:rns_gadget",
        "@shell_encryption//shell_encryption/rns:rns_galois_key",
        "@shell_encryption//shell_encryption/rns:rns_integer",
        "@shell_encryption//shell_encryption/rns:rns_modulus",
        "@shell_encryption//shell_encryption/rns:rns_polynomial",
        "@shell_encryption//shell_encryption/rns:rns_secret_key",
    ],
)

py_library(
    name = "shell_ops_py",
    srcs = [
        "python/shell_ops.py",
        "python/shell_optimizers.py",
    ],
    data = [
        ":python/_shell_ops.so",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "hadal_flow_lib",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/discrete_gaussian.py",
        "python/shell_context.py",
        "python/shell_key.py",
        "python/shell_tensor.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":shell_ops_py",
    ],
)

py_package(
    name = "hadal_flow_pkg",
    visibility = ["//:__pkg__"],
    deps = [
        ":hadal_flow_lib",
    ],
)
