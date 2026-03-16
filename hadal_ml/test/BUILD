load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "mnist_post_scale_test",
    size = "large",
    srcs = ["mnist_post_scale_test.py"],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "mnist_noenc_backprop_test",
    size = "large",
    srcs = ["mnist_noenc_backprop_test.py"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "mnist_enc_backprop_test",
    size = "large",
    srcs = ["mnist_enc_backprop_test.py"],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "dropout_test",
    size = "medium",
    srcs = ["dropout_test.py"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "embedding_test",
    size = "large",
    srcs = ["embedding_test.py"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "conv2d_test",
    size = "large",
    srcs = ["conv2d_test.py"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "max_pool2d_test",
    size = "medium",
    srcs = ["max_pool2d_test.py"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "dpsgd_model_local_test_maeloss",
    size = "enormous",
    srcs = [
        "dpsgd_model_local_test_maeloss.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "dpsgd_model_local_test",
    size = "enormous",
    srcs = [
        "dpsgd_model_local_test.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "dpsgd_model_float64_test",
    size = "enormous",
    srcs = [
        "dpsgd_model_float64_test.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "dpsgd_conv_model_local_test",
    size = "enormous",
    srcs = [
        "dpsgd_conv_model_local_test.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "dpsgd_model_distrib_test",
    size = "enormous",
    srcs = [
        "dpsgd_model_distrib_test.py",
        "test_models.py",
    ],
    env = {
        "HADAL_FILESYSTEM_PATH": "/tmp/hadal_model_distrib_test",
    },
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "postscale_model_local_test",
    size = "enormous",
    srcs = [
        "postscale_model_local_test.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "postscale_cnn_local_test",
    size = "enormous",
    srcs = [
        "postscale_cnn_local_test.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "postscale_model_distrib_test",
    size = "enormous",
    srcs = [
        "postscale_model_distrib_test.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "postscale_model_float64_test",
    size = "enormous",
    srcs = [
        "postscale_model_float64_test.py",
        "test_models.py",
    ],
    tags = ["exclusive"],
    deps = [
        "//hadal_ml",
        requirement("tensorflow"),
    ],
)
