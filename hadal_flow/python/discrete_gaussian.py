import tensorflow as tf
import hadal_flow.python.shell_ops as shell_ops


class DiscreteGaussianParams(tf.experimental.ExtensionType):
    max_scale: float
    base_scale: float


def sample_centered_gaussian_f(scale, params):
    return shell_ops.sample_centered_gaussian_f64(
        scale=scale, base_scale=params.base_scale, max_scale=params.max_scale
    )


def sample_centered_gaussian_l(context, num_samples, params):
    return shell_ops.sample_centered_gaussian_l64(
        context._raw_contexts[0],
        num_samples=num_samples,
        base_scale=params.base_scale,
        max_scale=params.max_scale,
    )
