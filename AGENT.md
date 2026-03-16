# Hadal Flow

## General Instructions

- Prefer to run tests frequently to catch syntax, compilation, and logic bugs early.
- If running tests fails because dependencies are not installed (e.g. bazel) the issue is likely because the command is not being run inside the devcontainer, a docker container which has all required dependencies already installed.
- The devcontainer is defined in ./.devcontainer, and can be started with the devcontainer cli. For example:
    ```bash
    devcontainer up --workspace-folder .
    devcontainer exec --workspace-folder . /bin/bash
    ```
- The tests can be run in a shell inside the devcontainer with:
    ```bash
    bazel test //hadal_flow/test:... //hadal_ml/test:...
    ```
- The devcontainer can be rebuilt with the command, however always stop and check with the user before doing so because it should not be necessary during normal development and something might be wrong.
    ```bash
    devcontainer up --workspace-folder . --remove-existing-container
    ```


## Reading Library Code

- Dependencies of this project are managed by bazel, which means they only exist in the filesystem inside the devcontainer.
- The SHELL encryption library code is used extensively in the C++ kernels and lives at the following path:
    `bazel-hadal_flow/external/_main~_repo_rules~shell_encryption/shell_encryption/`
- Reading these files requires using the devcontainer and running shell commands, for example:
    ```bash
    devcontainer up --workspace-folder .
    devcontainer exec --workspace-folder . /bin/bash
    cat ./bazel-hadal_flow/external/_main~_repo_rules~shell_encryption/shell_encryption/rns/rns_bgv_ciphertext.h
    ```

## Reading Test Logs

- Test logs are also inside the devcontainer's filesystem. If you see a path that looks like `/home/vscode/.cache/bazel/_bazel_vscode/8524201230cf289da6a8f50b894245cf/execroot/_main/bazel-out/k8-opt/testlogs/hadal_flow/test/shape_test/test.log` then it must be read from the devcontainer using the commands like those above.

