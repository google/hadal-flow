import tensorflow as tf
import hadal_flow as hadal


# These test cases are for the PtPtOptimizer, which optimizes the tf graph by
# reordering plaintext - plaintext operations which look like op( encode(a),
# <encode(b)>) to encode( op(a, <b>) ) where <> indicate optional arguments.

test_values_num_bits = 8


def gen_autocontext(cleartext_sz, noise_offset, scaling_factor=1):
    return hadal.create_autocontext64(
        log2_cleartext_sz=cleartext_sz,
        scaling_factor=scaling_factor,
        noise_offset_log2=noise_offset,
        noise_variance=8,
    )


def gen_context(scaling_factor=1):
    # Large moduli to avoid overflow in all test cases.
    return hadal.create_context64(
        log_n=11,
        main_moduli=[288230376151760897, 288230376152137729],
        plaintext_modulus=4294991873,
        scaling_factor=scaling_factor,
        seed="test_seed",
    )


@tf.function
def ct_ct_add(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits, 0) if use_auto_context else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)
    b = hadal.to_encrypted(cleartext_b, key, shell_context)

    intermediate = a + b
    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_ct_mul(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2, 0)
        if use_auto_context
        else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)
    b = hadal.to_encrypted(cleartext_b, key, shell_context)

    intermediate = a * b
    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_pt_add(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2, 0)
        if use_auto_context
        else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)
    b = hadal.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = a + b
    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_pt_mul(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2, 0)
        if use_auto_context
        else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)
    b = hadal.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = a * b
    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def long_arith(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits * 2, 0)
        if use_auto_context
        else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)
    b = hadal.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = ((a * b) + a) + b
    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def long_arith_with_scaling(cleartext_a, cleartext_b, use_auto_context=False):
    # To use scaling factor, inputs need to be floating point.
    cleartext_a = tf.cast(cleartext_a, tf.float32)
    cleartext_b = tf.cast(cleartext_b, tf.float32)

    scaling_factor = 3
    shell_context = (
        gen_autocontext(
            test_values_num_bits * 2 + 4 + scaling_factor.bit_length(),
            0,
            scaling_factor,
        )
        if use_auto_context
        else gen_context(scaling_factor)
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)
    b = hadal.to_shell_plaintext(cleartext_b, shell_context)

    intermediate = ((a * b) + a) + b
    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def reduce_sum_axis_1(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits + cleartext_a.shape[1].bit_length(), 0)
        if use_auto_context
        else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)

    intermediate = hadal.reduce_sum(a, axis=1)

    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def reduce_sum_axis_0(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits + cleartext_a.shape[0].bit_length(), 0)
        if use_auto_context
        else gen_context()
    )
    key = hadal.create_key64(shell_context)
    public_rotation_key = hadal.create_rotation_key64(shell_context, key)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)

    intermediate = hadal.reduce_sum(a, axis=0, rotation_key=public_rotation_key)

    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def fast_reduce_sum_axis_0(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits + cleartext_a.shape[0].bit_length(), 0)
        if use_auto_context
        else gen_context()
    )
    key = hadal.create_key64(shell_context)
    secret_fast_rotation_key = hadal.create_fast_rotation_key64(shell_context, key)

    a = hadal.to_encrypted(cleartext_a, key, shell_context)

    intermediate = hadal.fast_reduce_sum(a)

    result = hadal.to_tensorflow(intermediate, secret_fast_rotation_key)
    return result


@tf.function
def ct_roll(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits, 0) if use_auto_context else gen_context()
    )
    key = hadal.create_key64(shell_context)
    public_rotation_key = hadal.create_rotation_key64(shell_context, key)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)

    intermediate = hadal.roll(a, 5, rotation_key=public_rotation_key)

    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_expand_dims(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits, 0) if use_auto_context else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)

    intermediate = hadal.expand_dims(a, axis=1)

    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_concat(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context = (
        gen_autocontext(test_values_num_bits, 0) if use_auto_context else gen_context()
    )
    key = hadal.create_key64(shell_context)
    a = hadal.to_encrypted(cleartext_a, key, shell_context)
    b = hadal.to_encrypted(cleartext_b, key, shell_context)

    intermediate = hadal.concat([a, b], axis=1)

    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def multi_context(cleartext_a, cleartext_b, use_auto_context=False):
    shell_context1 = (
        gen_autocontext(test_values_num_bits * 2, 10)  # Add noise so log_n matched
        if use_auto_context
        else gen_context()
    )
    key1 = hadal.create_key64(shell_context1)
    a1 = hadal.to_encrypted(cleartext_a, key1, shell_context1)
    b1 = hadal.to_encrypted(cleartext_b, key1, shell_context1)

    intermediate1 = a1 * b1
    result1 = hadal.to_tensorflow(intermediate1, key1)

    shell_context2 = (
        gen_autocontext(test_values_num_bits * 4, 0)
        if use_auto_context
        else gen_context()
    )  # Use an offset of 1 so the contexts are different and can't be shared.
    key2 = hadal.create_key64(shell_context2)
    a2 = hadal.to_encrypted(cleartext_a, key2, shell_context2)
    b2 = hadal.to_encrypted(cleartext_b, key2, shell_context2)

    intermediate2 = a2 * b2
    result2 = hadal.to_tensorflow(intermediate2, key2)

    return result1 + result2


def count_ops(graph, op_name):
    num_ct_pt_ops = 0
    for node in graph.as_graph_def().node:
        if node.op == op_name:
            num_ct_pt_ops += 1
    return num_ct_pt_ops


class TestAutoParamOptimizer(tf.test.TestCase):

    def _test_func(self, tf_func, num_autocontexts=1):
        shape = [100, 12]
        a = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=2**test_values_num_bits - 1,
        )
        b = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=2**test_values_num_bits - 1,
        )
        a = tf.cast(a, tf.uint64)
        b = tf.cast(b, tf.uint64)

        func = tf_func.get_concrete_function(a, b, True)

        # print("\noriginal graph:")
        # for node in func.graph.as_graph_def().node:
        #     print(f"{node.name} {node.op}({node.input})")

        orig_num_auto_ops = count_ops(func.graph, "AutoShellContext64")
        self.assertEqual(orig_num_auto_ops, num_autocontexts)

        # Optimize the graph using hadal HE-specific optimizers.
        optimized_func = hadal.optimize_shell_graph(func, ["ModuliAutotuneOptimizer"])

        # print("\noptimized graph:")
        # for node in optimized_func.graph.as_graph_def().node:
        #     print(f"{node.name} {node.op}({node.input})")

        # Call the optimized function.
        c = optimized_func(a, b, True)
        # Can remove pack_output above if
        # https://github.com/tensorflow/tensorflow/pull/67612 is merged.
        c = optimized_func.function_type.pack_output(c)

        opt_num_auto_ops = count_ops(optimized_func.graph, "AutoShellContext64")
        self.assertEqual(opt_num_auto_ops, 0)
        opt_num_ctx_ops = count_ops(optimized_func.graph, "ContextImport64")
        self.assertEqual(opt_num_ctx_ops, num_autocontexts)

        # Check the optimized graph still computes the correct value.
        eager_c = tf_func(a, b, False)

        # c and eager c may be different dimensions due to the number of slots
        # chosen by the optimizer. Match the dimensions before comparing the
        # values.
        if tf_func == reduce_sum_axis_0 or tf_func == fast_reduce_sum_axis_0:
            # Concatenate the first and middle slots of each value before comparing.
            eager_c = tf.concat([eager_c[0], eager_c[eager_c.shape[0] // 2]], axis=0)
            c = tf.concat([c[0], c[c.shape[0] // 2]], axis=0)

        else:
            # Pad the first (slotting) dimension of the outputs to the same value.
            def pad_first_dim(tensor, first_dim):
                if first_dim > tensor.shape[0]:
                    return tf.pad(
                        tensor,
                        [[0, first_dim - tensor.shape[0]]]
                        + [[0, 0] for _ in range(len(tensor.shape) - 1)],
                    )
                else:
                    return tensor

            max_fist_dim = tf.maximum(c.shape[0], eager_c.shape[0])
            eager_c = pad_first_dim(eager_c, max_fist_dim)
            c = pad_first_dim(c, max_fist_dim)

        self.assertAllEqual(c, eager_c)

    def test_func(self):
        with self.subTest(f"Optimizer for func ct_ct_add."):
            self._test_func(ct_ct_add)

        with self.subTest(f"Optimizer for func ct_ct_mul."):
            self._test_func(ct_ct_mul)

        with self.subTest(f"Optimizer for func ct_pt_add."):
            self._test_func(ct_pt_add)

        with self.subTest(f"Optimizer for func ct_pt_mul."):
            self._test_func(ct_pt_mul)

        with self.subTest(f"Optimizer for func long_arith."):
            self._test_func(long_arith)

        with self.subTest(f"Optimizer for func long_arith_with_scaling."):
            self._test_func(long_arith_with_scaling)

        with self.subTest(f"Optimizer for reduce sum axis 1."):
            self._test_func(reduce_sum_axis_1)

        with self.subTest(f"Optimizer for reduce sum axis 0."):
            self._test_func(reduce_sum_axis_0)

        with self.subTest(f"Optimizer for fast reduce sum axis 0."):
            self._test_func(fast_reduce_sum_axis_0)

        with self.subTest(f"Optimizer for roll."):
            self._test_func(ct_roll)

        with self.subTest(f"Optimizer for expand_dims."):
            self._test_func(ct_expand_dims)

        with self.subTest(f"Optimizer for concat."):
            self._test_func(ct_concat)

        with self.subTest(f"Optimizer for multi context."):
            self._test_func(multi_context, num_autocontexts=2)


class TestAutoParamEnableOptimizer(tf.test.TestCase):
    def test_func(self):
        shape = [20, 1]
        a = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=127,
        )
        b = tf.random.uniform(
            shape,
            dtype=tf.int64,
            minval=0,
            maxval=127,
        )

        hadal.enable_optimization()
        c = ct_ct_add(a, b, True)

        # If the optimizer ran, the shape should be padded out to the
        # ciphertext modulus.
        self.assertNotEqual(c.shape, shape)


if __name__ == "__main__":
    tf.test.main()
