"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# This file is used to rename the wheel generated via py_wheel with values
# determined at runtime
import sys, os, re, platform
from packaging import tags, version
from platform import python_version
from enum import Enum


bazel_dir_var = "BUILD_WORKSPACE_DIRECTORY"
if not bazel_dir_var in os.environ:
    raise Exception(f"{bazel_dir_var} not defined")
build_workspace_dir = os.environ[bazel_dir_var]
bazel_bin_dir = build_workspace_dir + "/bazel-bin/"
whl_prefix = "hadal_flow"


class ABIVersion(Enum):
    MANY_LINUX_X_Y = "manylinux_" + str(platform.libc_ver()[1].replace(".", "_"))
    MANY_LINUX_2014 = "manylinux2014"
    MANY_LINUX_2010 = "manylinux2010"


# We rely on the bundled pip versions installed with python to determine the
# naming variant: https://github.com/pypa/manylinux
#
# NOTE: Order matters
python_versions = {
    version.parse("3.10.0"): ABIVersion.MANY_LINUX_X_Y,
    version.parse("3.9.5"): ABIVersion.MANY_LINUX_X_Y,
    version.parse("3.9.0"): ABIVersion.MANY_LINUX_2014,
    version.parse("3.8.10"): ABIVersion.MANY_LINUX_X_Y,
    version.parse("3.8.4"): ABIVersion.MANY_LINUX_2014,
    version.parse("3.8.0"): ABIVersion.MANY_LINUX_2010,  # Fails for aarch64
}


inputfile = ""
for file in os.listdir(bazel_bin_dir):
    if (
        file.startswith(whl_prefix + "-")
        and file.find("INTERPRETER") > 0
        and file.endswith(".whl")
    ):
        inputfile = os.path.join(bazel_bin_dir, file)
        break

if not inputfile:
    print(f"ERROR: wheel not found in {bazel_bin_dir}")
    sys.exit(1)

print(f"found wheel: {inputfile}")

interpreter_name = tags.interpreter_name()
interpreter_version = tags.interpreter_version()
abi_tag = interpreter_name + interpreter_version

# File looks like
# <whl_prefix>-0.0.1-INTERPRETER-ABI-PLATFORM-ARCH.whl
# openmined.psi-1.0.0-INTERPRETER-ABI-PLATFORM-ARCH.whl
# INTERPRETER and ABI should be the same value
outfile = re.sub(r"INTERPRETER", abi_tag, inputfile)
outfile = re.sub(r"ABI", abi_tag, outfile)
system = platform.system()

# We rename the wheel depending on the version of python and glibc
if system.lower() == "linux":
    current_version = version.parse(python_version())
    manylinux = ABIVersion.MANY_LINUX_X_Y.value
    for ver, ml in python_versions.items():
        if current_version >= ver:
            manylinux = ml.value
            break

    outfile = re.sub(r"LINUX", manylinux, outfile)

print("renaming ", inputfile, outfile)
os.replace(inputfile, outfile)
