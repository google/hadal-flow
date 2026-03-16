import tensorflow as tf
import hadal_flow as hadal
import test_utils


# These test cases are for the PtPtOptimizer, which optimizes the tf graph by
# reordering plaintext - plaintext operations which look like op( encode(a),
# <encode(b)>) to encode( op(a, <b>) ) where <> indicate optional arguments.


@tf.function
def pt_add(a, b, num_opts, shell_context, key):
    a = hadal.to_shell_plaintext(a, shell_context)
    b = hadal.to_shell_plaintext(b, shell_context)
    intermediate = a
    for _ in range(num_opts):
        intermediate += b
    result = hadal.to_tensorflow(intermediate)
    return result


@tf.function
def pt_sub(a, b, num_opts, shell_context, key):
    a = hadal.to_shell_plaintext(a, shell_context)
    b = hadal.to_shell_plaintext(b, shell_context)
    intermediate = a
    for _ in range(num_opts):
        intermediate -= b
    result = hadal.to_tensorflow(intermediate)
    return result


@tf.function
def pt_neg(a, b, num_opts, shell_context, key):
    a = hadal.to_shell_plaintext(a, shell_context)
    intermediate = a
    for _ in range(num_opts):
        intermediate = -intermediate
    result = hadal.to_tensorflow(intermediate)
    return result


@tf.function
def pt_mul(a, b, num_opts, shell_context, _):
    a = hadal.to_shell_plaintext(a, shell_context)
    b = hadal.to_shell_plaintext(b, shell_context)
    intermediate = a
    for _ in range(num_opts):
        intermediate *= b
    result = hadal.to_tensorflow(intermediate)
    return result


@tf.function
def ct_add_no_opt(a, b, num_opts, shell_context, key):
    # This should not be optimized, it is not a plaintext operation.
    ct_a = hadal.to_encrypted(a, key, shell_context)
    ct_b = hadal.to_encrypted(b, key, shell_context)

    intermediate = ct_a
    for _ in range(num_opts):
        intermediate += ct_b

    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_sub_no_opt(a, b, num_opts, shell_context, key):
    # This should not be optimized, it is not a plaintext operation.
    ct_a = hadal.to_encrypted(a, key, shell_context)
    ct_b = hadal.to_encrypted(b, key, shell_context)

    intermediate = ct_a
    for _ in range(num_opts):
        intermediate -= ct_b

    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def ct_mul_no_opt(a, b, num_opts, shell_context, key):
    # This should not be optimized, it is not a plaintext operation.
    ct_a = hadal.to_encrypted(a, key, shell_context)
    ct_b = hadal.to_encrypted(b, key, shell_context)

    intermediate = ct_a
    for _ in range(num_opts):
        intermediate *= ct_b
    result = hadal.to_tensorflow(intermediate, key)
    return result


@tf.function
def pt_enc_dec(a, b, num_opts, shell_context, key):
    # The encode decode pairs should be optimized away.
    result = a
    for _ in range(num_opts):
        pt_a = hadal.to_shell_plaintext(result, shell_context)
        aa = hadal.to_tensorflow(pt_a)
        result += aa

    return result


@tf.function
def pt_enc_dec_complex(a, b, num_opts, shell_context, key):
    # The encode decode pairs should be optimized away even when the graph
    # output is a decode and there are intermediate ops between encode and
    # decode.
    pt_a = hadal.to_shell_plaintext(a, shell_context)
    result = pt_a
    for _ in range(num_opts - 1):
        result += pt_a
        result = hadal.to_tensorflow(result)
        result = hadal.to_shell_plaintext(result, shell_context)

    result = hadal.to_tensorflow(result)
    return result


def count_ops(graph, op_name):
    num_ct_pt_ops = 0
    for node in graph.as_graph_def().node:
        if node.op == op_name:
            num_ct_pt_ops += 1
    return num_ct_pt_ops


class TestPtPtOptimizer(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        int_dtypes = [
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.uint32,
            tf.int32,
            tf.uint64,
            tf.int64,
        ]
        cls.test_contexts = []

        cls.test_contexts.append(
            test_utils.TestContext(
                outer_shape=[1],
                plaintext_dtype=tf.float32,
                log_n=11,
                main_moduli=[8556589057, 8388812801],
                aux_moduli=[],
                plaintext_modulus=40961,
                scaling_factor=1,
            )
        )

    def _test_func(
        self, test_context, tf_func, num_pt_ops, expected_num_pt_ops, pt_op_name
    ):
        a = test_utils.uniform_for_n_muls(test_context, num_pt_ops)
        b = test_utils.uniform_for_n_muls(test_context, num_pt_ops)

        func = tf_func.get_concrete_function(
            a, b, num_pt_ops, test_context.shell_context, test_context.key
        )
        orig_num_ops = count_ops(func.graph, pt_op_name)
        self.assertEqual(orig_num_ops, num_pt_ops)

        # print("\noriginal graph:")
        # for node in func.graph.as_graph_def().node:
        #     print(f"{node.name} {node.op}({node.input})", flush=True)

        # Optimize the graph using hadal HE-specific optimizers.
        optimized_func = hadal.optimize_shell_graph(func)
        # Call the optimized function.
        c = optimized_func(
            a, b, num_pt_ops, test_context.shell_context, test_context.key
        )
        # Can remove pack_output above if
        # https://github.com/tensorflow/tensorflow/pull/67612 is merged.
        c = optimized_func.function_type.pack_output(c)
        opt_num_ops = count_ops(optimized_func.graph, pt_op_name)

        # print("\noptimized graph:")
        # for node in optimized_func.graph.as_graph_def().node:
        #     print(f"{node.name} {node.op}({node.input})", flush=True)

        self.assertEqual(opt_num_ops, expected_num_pt_ops)

        # Check the optimized graph still computes the correct value.
        self.assertAllClose(
            c,
            tf_func(a, b, num_pt_ops, test_context.shell_context, test_context.key),
            atol=1 / test_context.shell_context.scaling_factor * num_pt_ops,
        )

    def test_func(self):
        for test_context in self.test_contexts:
            with self.subTest(f"Optimizer for func pt_add."):
                self._test_func(test_context, pt_add, 7, 0, "AddPtPt64")

            with self.subTest(f"Optimizer for func pt_sub."):
                self._test_func(test_context, pt_sub, 7, 0, "SubPtPt64")

            with self.subTest(f"Optimizer for func pt_mul."):
                self._test_func(test_context, pt_mul, 2, 0, "MulPtPt64")

            with self.subTest(f"Optimizer for func pt_neg."):
                self._test_func(test_context, pt_neg, 1, 0, "NegPt64")

            with self.subTest(f"Optimizer for func ct_add_no_opt."):
                self._test_func(test_context, ct_add_no_opt, 7, 7, "AddCtCt64")

            with self.subTest(f"Optimizer for func ct_sub_no_opt."):
                self._test_func(test_context, ct_sub_no_opt, 7, 7, "SubCtCt64")

            with self.subTest(f"Optimizer for func ct_mul_no_opt."):
                self._test_func(test_context, ct_mul_no_opt, 1, 1, "MulCtCt64")

            with self.subTest(f"Optimizer for func pt_enc_dec."):
                self._test_func(test_context, pt_enc_dec, 4, 0, "PolynomialImport64")

            with self.subTest(f"Optimizer for func pt_enc_dec_complex."):
                self._test_func(
                    test_context, pt_enc_dec_complex, 4, 0, "PolynomialImport64"
                )


class TestPtPtAutoEnableOptimizer(tf.test.TestCase):
    test_contexts = None

    @classmethod
    def setUpClass(cls):
        int_dtypes = [
            tf.uint8,
            tf.int8,
            tf.uint16,
            tf.int16,
            tf.uint32,
            tf.int32,
            tf.uint64,
            tf.int64,
        ]
        cls.test_contexts = []

        cls.test_contexts.append(
            test_utils.TestContext(
                outer_shape=[1],
                plaintext_dtype=tf.float32,
                log_n=11,
                main_moduli=[8556589057, 8388812801],
                aux_moduli=[],
                plaintext_modulus=40961,
                scaling_factor=1,
            )
        )

    def test_auto_optimize(self):
        from timeit import timeit

        test_context = self.test_contexts[0]

        a = test_utils.uniform_for_n_adds(test_context, 10)
        b = test_utils.uniform_for_n_adds(test_context, 10)

        # Call the function as usual.
        unopt_time = timeit(
            lambda: pt_add(a, b, 10, test_context.shell_context, test_context.key),
            number=10,
        )

        # Turn on automatic optimization. Note there is no way to get the
        # optimized graph from the tf.function so we need to rely on timing info
        # to make sure it's turned on.
        hadal.enable_optimization(["PtPtOptimizer"])

        opt_time = timeit(
            lambda: pt_add(a, b, 10, test_context.shell_context, test_context.key),
            number=10,
        )

        # Optimized time should be faster.
        self.assertLess(opt_time, unopt_time * 0.6)


if __name__ == "__main__":
    tf.test.main()
