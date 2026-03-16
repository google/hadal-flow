load("@buildifier_prebuilt//:rules.bzl", "buildifier")
load("@pip//:requirements.bzl", "requirement")
load("@python_versions//3.10:defs.bzl", compile_pip_requirements_3_10 = "compile_pip_requirements")
load("@python_versions//3.11:defs.bzl", compile_pip_requirements_3_11 = "compile_pip_requirements")
load("@python_versions//3.12:defs.bzl", compile_pip_requirements_3_12 = "compile_pip_requirements")
load("@python_versions//3.13:defs.bzl", compile_pip_requirements_3_13 = "compile_pip_requirements")
load("@python_versions//3.9:defs.bzl", compile_pip_requirements_3_9 = "compile_pip_requirements")
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:packaging.bzl", "py_wheel")

DEFAULT_PYTHON = "3.10"  # Also see ./MODULE.bazel

exports_files([
    "LICENSE",
    "setup.py",
    "requirements.in",
    "requirements_3_9.txt",
    "requirements_3_10.txt",
    "requirements_3_11.txt",
    "requirements_3_12.txt",
    "requirements_3_13.txt",
    "README.md",
    "DESCRIPTION.md",
])

compile_pip_requirements_3_9(
    name = "requirements_3_9",
    extra_args = ["--allow-unsafe"],  # need setuptools
    requirements_in = "//:requirements.in",
    requirements_txt = "//:requirements_3_9.txt",
    visibility = ["//visibility:public"],
)

compile_pip_requirements_3_10(
    name = "requirements_3_10",
    extra_args = ["--allow-unsafe"],  # need setuptools
    requirements_in = "//:requirements.in",
    requirements_txt = "//:requirements_3_10.txt",
    visibility = ["//visibility:public"],
)

compile_pip_requirements_3_11(
    name = "requirements_3_11",
    extra_args = ["--allow-unsafe"],  # need setuptools
    requirements_in = "//:requirements.in",
    requirements_txt = "//:requirements_3_11.txt",
    visibility = ["//visibility:public"],
)

compile_pip_requirements_3_12(
    name = "requirements_3_12",
    extra_args = ["--allow-unsafe"],  # need setuptools
    requirements_in = "//:requirements.in",
    requirements_txt = "//:requirements_3_12.txt",
    visibility = ["//visibility:public"],
)

compile_pip_requirements_3_13(
    name = "requirements_3_13",
    extra_args = ["--allow-unsafe"],  # need setuptools
    requirements_in = "//:requirements.in",
    requirements_txt = "//:requirements_3_13.txt",
    visibility = ["//visibility:public"],
)

buildifier(
    name = "bazel_formatter",
    exclude_patterns = [
        "./bazel-*/*",
        "./.git/*",
    ],
    lint_mode = "warn",
)

py_binary(
    name = "python_formatter",
    srcs = ["//tools:python_formatter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//:__pkg__",
    ],
    deps = [
        requirement("black"),
    ],
)

py_binary(
    name = "clang_formatter",
    srcs = ["//tools:clang_formatter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//:__pkg__",
    ],
)

py_binary(
    name = "wheel_rename",
    srcs = ["//tools:wheel_rename.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//:__pkg__",
    ],
    deps = [
        requirement("packaging"),
    ],
)

py_wheel(
    name = "wheel",
    abi = "ABI",
    author = "Google Inc.",
    author_email = "jchoncholas@gmail.com",
    classifiers = [
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    description_file = "//:README.md",
    distribution = module_name(),
    extra_distinfo_files = {
        "//:LICENSE": "LICENSE",
        "//:README.md": "README",
    },
    homepage = "https://github.com/google/hadal-flow",
    license = "Apache 2.0",
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:windows_arm64": "win_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_12_0_x86_64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_12_0_arm64",
        "@bazel_tools//src/conditions:linux_x86_64": "LINUX_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": "LINUX_aarch64",
    }),
    python_requires = ">=3.9",
    python_tag = "INTERPRETER",
    requires = ["tensorflow[and-cuda]==2.20.0"],  # See also: requirements.in.
    # The summary is tailored for each python version because PyPI prevents
    # wheel uploads for different versions which have the same contents.
    # Changing the summary is sufficient to allow re-uploads.
    summary = "Hadal: Privacy preserving machine learning with Tensorflow and the SHELL encryption library, built for python " + DEFAULT_PYTHON + ".",
    version = module_version(),
    deps = [
        "//hadal_flow:hadal_flow_pkg",
        "//hadal_ml:hadal_ml_pkg",
    ],
)
