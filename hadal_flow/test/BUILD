load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "context_test",
    size = "small",
    srcs = [
        "context_test.py",
    ],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "basic_test",
    size = "small",
    srcs = [
        "basic_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "scaling_factor_test",
    size = "small",
    srcs = [
        "scaling_factor_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "add_test",
    size = "medium",
    srcs = [
        "add_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "mul_test",
    size = "medium",
    srcs = [
        "mul_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "mat_mul_test",
    size = "medium",
    srcs = [
        "mat_mul_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "composite_test",
    size = "small",
    srcs = [
        "composite_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "rotation_test",
    size = "large",
    srcs = [
        "rotation_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "rotation_test_fast",
    size = "large",
    srcs = [
        "rotation_test_fast.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "shape_test",
    size = "medium",
    srcs = [
        "shape_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "key_test",
    size = "small",
    srcs = [
        "key_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "segment_test",
    size = "enormous",
    srcs = [
        "segment_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "conv2d_test",
    size = "medium",
    srcs = [
        "conv2d_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "ct_pt_optimizer_test",
    size = "medium",
    srcs = [
        "ct_pt_optimizer_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "random_round_test",
    size = "small",
    srcs = [
        "random_round_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "pt_pt_optimizer_test",
    size = "medium",
    srcs = [
        "pt_pt_optimizer_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "auto_param_optimizer_test",
    size = "medium",
    srcs = [
        "auto_param_optimizer_test.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "clip_and_noise_test",
    size = "medium",
    srcs = [
        "clip_and_noise_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "distribution_test",
    size = "medium",
    srcs = [
        "distribution_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "filesystem_distribution_test",
    size = "medium",
    srcs = [
        "filesystem_distribution_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "discrete_gaussian_test",
    size = "small",
    srcs = [
        "discrete_gaussian_test.py",
        "test_utils.py",
    ],
    imports = ["./"],
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "large_tensor_test_fs",
    size = "large",
    srcs = [
        "large_tensor_test.py",
    ],
    env = {
        "HADAL_FILESYSTEM_PATH": "/tmp/hadal_large_test_fs",
    },
    imports = ["./"],
    main = "large_tensor_test.py",
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "large_tensor_test_grpc",
    size = "large",
    srcs = [
        "large_tensor_test.py",
    ],
    imports = ["./"],
    main = "large_tensor_test.py",
    deps = [
        "//hadal_flow:hadal_flow_lib",
        requirement("tensorflow"),
    ],
)
