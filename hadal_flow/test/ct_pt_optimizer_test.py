import tensorflow as tf
import hadal_flow as hadal
import test_utils


# These test cases are for the CtPtOptimizer, which optimizes the tf graph by
# reordering ciphertext - plaintext operations which look like
# ((ct + pt) + pt) to (ct + (pt + pt).


@tf.function
def ct_pt_pt_add(ct, pt, num_opts):
    intermediate = ct
    for _ in range(num_opts):
        intermediate += pt
    return intermediate


@tf.function
def ct_pt_pt_sub(ct, pt, num_opts):
    intermediate = ct
    for _ in range(num_opts):
        intermediate -= pt
    return intermediate


@tf.function
def ct_pt_pt_add_sub(ct, pt, num_opts):
    intermediate = ct
    for i in range(num_opts):
        if i % 2 == 0:
            intermediate += pt
        else:
            intermediate -= pt
    return intermediate


@tf.function
def ct_pt_pt_mul(ct, pt, num_opts):
    intermediate = ct
    for _ in range(num_opts):
        intermediate *= pt
    return intermediate


@tf.function
def ct_pt_pt_add_mul_no_opt(ct, pt, num_opts):
    # This should not be optimized, mul and add are not associative.
    intermediate = ct + pt
    for _ in range(num_opts):
        intermediate *= pt
    return intermediate


@tf.function
def ct_pt_pt_mul_add_no_opt(ct, pt, num_opts):
    # This should not be optimized, mul and add are not associative.
    intermediate = ct * pt
    for _ in range(num_opts):
        intermediate += pt
    return intermediate


@tf.function
def ct_pt_pt_reused_no_opt(ct, pt, num_opts):
    inner = ct + pt
    intermediate = inner
    for _ in range(num_opts - 1):
        intermediate += pt
    # At this point the graph could be optimized to (ct + (pt + pt + ...))

    result = intermediate + inner
    # Now since `inner` is re-used, the first op should not be optimized.
    return result


def count_ct_pt_ops(graph, op_name):
    num_ct_pt_ops = 0
    for node in graph.as_graph_def().node:
        if node.op == op_name:
            num_ct_pt_ops += 1
    return num_ct_pt_ops


class TestCtPtOptimizer(tf.test.TestCase):
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
        a = test_utils.uniform_for_n_muls(test_context, num_pt_ops + 2)
        b = test_utils.uniform_for_n_muls(test_context, num_pt_ops + 2)

        ct_a = hadal.to_encrypted(a, test_context.key, test_context.shell_context)
        pt_b = hadal.to_shell_plaintext(b, test_context.shell_context)

        # Sanity check the plain TensorFlow function correctly computes the
        # correct value.
        enc_c = tf_func(ct_a, pt_b, num_pt_ops)
        self.assertAllClose(
            hadal.to_tensorflow(enc_c, test_context.key),
            tf_func(a, b, num_pt_ops),
            atol=1 / test_context.shell_context.scaling_factor * num_pt_ops,
        )

        func = tf_func.get_concrete_function(ct_a, pt_b, num_pt_ops)
        orig_num_ops = count_ct_pt_ops(func.graph, pt_op_name)
        self.assertEqual(orig_num_ops, num_pt_ops)

        # print("\noriginal graph:")
        # for node in func.graph.as_graph_def().node:
        #     print(f'{node.name} {node.op}({node.input})')

        # Optimize the graph using hadal HE-specific optimizers.
        optimized_func = hadal.optimize_shell_graph(func)
        # Call the optimized function.
        enc_c = optimized_func(ct_a, pt_b, num_pt_ops)
        # Can remove pack_output above if
        # https://github.com/tensorflow/tensorflow/pull/67612 is merged.
        enc_c = optimized_func.function_type.pack_output(enc_c)
        opt_num_ops = count_ct_pt_ops(optimized_func.graph, pt_op_name)

        # print("\noptimized graph:")
        # for node in optimized_func.graph.as_graph_def().node:
        #     print(f'{node.name} {node.op}({node.input})')

        self.assertEqual(opt_num_ops, expected_num_pt_ops)

        # Check the optimized graph still computes the correct value.
        self.assertAllClose(
            hadal.to_tensorflow(enc_c, test_context.key),
            tf_func(a, b, num_pt_ops),
            atol=1 / test_context.shell_context.scaling_factor * num_pt_ops,
        )

    def test_func(self):
        for test_context in self.test_contexts:
            with self.subTest(f"Optimizer for func ct_pt_pt_add."):
                self._test_func(test_context, ct_pt_pt_add, 7, 1, "AddCtPt64")

            with self.subTest(f"Optimizer for func ct_pt_pt_sub."):
                self._test_func(test_context, ct_pt_pt_sub, 7, 1, "SubCtPt64")

            with self.subTest(f"Optimizer for func ct_pt_pt_mul."):
                self._test_func(test_context, ct_pt_pt_mul, 2, 1, "MulCtPt64")

            with self.subTest(f"Optimizer for func ct_pt_pt_add_mul_no_opt."):
                # Ensure both the add and mul CtPt operations remain in the
                # graph.
                self._test_func(
                    test_context, ct_pt_pt_add_mul_no_opt, 1, 1, "MulCtPt64"
                )
                self._test_func(
                    test_context, ct_pt_pt_add_mul_no_opt, 1, 1, "AddCtPt64"
                )

            with self.subTest(f"Optimizer for func ct_pt_pt_mul_add_no_opt."):
                # Ensure both the add and mul CtPt operations remain in the
                # graph.
                self._test_func(
                    test_context, ct_pt_pt_mul_add_no_opt, 1, 1, "MulCtPt64"
                )
                self._test_func(
                    test_context, ct_pt_pt_mul_add_no_opt, 1, 1, "AddCtPt64"
                )

            with self.subTest(f"Optimizer for func ct_pt_pt_reused_no_opt."):
                self._test_func(test_context, ct_pt_pt_reused_no_opt, 5, 2, "AddCtPt64")


class TestCtPtAutoEnableOptimizer(tf.test.TestCase):
    def test_auto_optimize(self):
        from timeit import timeit

        context = hadal.create_context64(
            log_n=10,
            main_moduli=[8556589057, 8388812801],
            plaintext_modulus=40961,
            scaling_factor=3,
            seed="test_seed",
        )

        secret_key = hadal.create_key64(context)

        a = tf.random.uniform([context.num_slots, 40000], dtype=tf.float32, maxval=10)
        b = tf.random.uniform([context.num_slots, 40000], dtype=tf.float32, maxval=10)

        ct_a = hadal.to_encrypted(a, secret_key, context)
        pt_b = hadal.to_shell_plaintext(b, context)

        # Call the function as usual.
        unopt_time = timeit(lambda: ct_pt_pt_add(ct_a, pt_b, 10), number=10)

        # Turn on automatic optimization. Note there is no way to get the
        # optimized graph from the tf.function so we need to rely on timing info
        # to make sure it's turned on.
        hadal.enable_optimization(["CtPtOptimizer"])

        opt_time = timeit(lambda: ct_pt_pt_add(ct_a, pt_b, 10), number=10)

        # Optimized time should be twice as fast due to the two ciphertext
        # components, but give it some slack and check it is faster.
        self.assertLess(opt_time, unopt_time * 0.6)


if __name__ == "__main__":
    tf.test.main()
