# hadal-flow

[![Build](https://github.com/google/hadal-flow/actions/workflows/wheel.yaml/badge.svg)](https://github.com/google/hadal-flow/actions/workflows/wheel.yaml)
[![Formatting](https://github.com/google/hadal-flow/actions/workflows/formatter.yaml/badge.svg)](https://github.com/google/hadal-flow/actions/workflows/formatter.yaml)

Train ML models with strong privacy.

`hadal_flow` is a TensorFlow extension that uses Homomorphic Encryption (HE) to
train models with centralized label differential privacy (Label DP) guarantees,
without requiring a trusted third party.

It's built for the "vertically partitioned" scenario where one party has the
features and another party has the labels. This library implements the protocols
from the Hadal research paper  to securely train a model without the
feature-holder ever seeing the plain-text labels.

This is not an officially supported Google product.

## Getting Started

```bash
pip install hadal-flow
```

See `./examples/` for how to use the library.

## The Problem: Centralized DP without a Trusted Party

When training with high privacy requirements (e.g., $\epsilon \le 1$),
centralized DP (like DP-SGD) provides much higher model accuracy than local DP
(like randomized response). However, centralized DP traditionally requires a
trusted third-party "curator" that can see all the data (or at least the secret
labels) to compute and noise the gradients. This is a problem when data is
vertically partitioned: Party F has the features and wants to train the model.
Party L has the secret labels and cannot share them. How can Party F train its
model using Party L's labels without a trusted intermediary?

## The Solution: HE-based Backpropagation

`hadal-flow` uses Homomorphic Encryption (via Google's
[SHELL](https://github.com/google/shell-encryption/) library) to
cryptographically "simulate" the trusted curator. The core technical idea is
based on the Features-And-Model-vs-Labels (FAML) data partitioning:
1. Forward Pass (Plaintext): Party F (with features and the model) computes the
entire forward pass in plaintext, right up to the final layer's logits.
2. Encrypted Labels: Party L encrypts its batch of labels using HE and sends the
single ciphertext to Party F.
3. Backward Pass (Encrypted): The gradient of the loss (e.g., CCE with Softmax)
is often a simple affine function of the labels (like $\hat{y} - y$). Party F
can compute this step homomorphically using its plaintext logits and Party L's
encrypted labels.
4. Model Update: Party F finishes the backpropagation, adds the required DP
noise, and updates its model weights.

The result is a model trained with the high utility of centralized DP, but Party
F never sees Party L's individual labels.

## What's Included?

The library is split into two packages:

- hadal_flow: The base package. It integrates TensorFlow with the SHELL library,
providing a ShellTensor type for basic HE-enabled computations.
- hadal_ml: The machine learning library. It implements two different
protocols for the encrypted backpropagation step:
  - POSTSCALE: A novel protocol that is highly efficient for models with a low
  number of output classes (e.g., binary classification).
  - HE-DP-SGD: A more direct HE implementation of backpropagation, which is
  better suited for models with many output classes.

## Building

### Build From Source

1. Install bazel and python3 or use the devcontainer.

2. Run the tests.

    ```bash
    bazel test //hadal_flow/...
    bazel test //hadal_ml/...  # Large tests, requires 128GB of memory.
    ```

3. Build the code.

    ```bash
    bazel build //:wheel
    bazel run //:wheel_rename
    ```

4. (Optional) Install the wheel, e.g. to try out the `./examples/`.
    You may first need to copy the wheel out of the devcontainer's filesystem.

    ```bash
    cp -f bazel-bin/*.whl ./  # Run in devcontainer if using.
    ```

    Then install.

    ```bash
    pip install --force-reinstall hadal_flow-*.whl  # Run in target environment.
    ```

Note the cpython api is not compatible across minor python versions (e.g. 3.10,
3.11) so the wheel must be rebuilt for each python version.

### Code Formatters and Counters

```bash
bazel run //:bazel_formatter
bazel run //:python_formatter
bazel run //:clang_formatter
```

```bash
cloc ./ --fullpath --not-match-d='/(bazel-.*|.*\.venv)/'
```

### Update Python Dependencies

Update requirements.in and run the following to update the requirements files
for each python version.

```bash
for ver in 3_9 3_10 3_11 3_12 3_13; do
  rm requirements_${ver}.txt
  touch requirements_${ver}.txt
  bazel run //:requirements_${ver}.update
done

bazel clean --expunge
```

If updating the tensorflow dependency, other dependencies may also need to
change, e.g. abseil (see `MODULE.bazel`). This issue usually manifests as a
missing symbols error in the tests when trying to import the tensorflow DSO. In
this case, `c++filt` will help to decode the mangled symbol name and `nm
--defined-only .../libtensorflow_framework.so | grep ...` may help find what the
symbol changed to, and which dependency is causing the error.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

Convolutions on AMD-based platforms may fail due to known limitations of
TensorFlow. This will resulting in the following error when running tests:

```log
CPU implementation of Conv3D currently only supports dilated rates of 1.
```

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
