# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import tensorflow as tf
import keras
import numpy as np
import hadal_flow as hadal
import hadal_ml

# # Num plaintext bits: 32, noise bits: 84
# # Max representable value: 654624
# context = hadal.create_context64(
#     log_n=11,
#     main_moduli=[288230376151748609, 144115188076060673],
#     plaintext_modulus=4294991873,
#     scaling_factor=3,
#     seed="test_seed",
# )
# 61 bits of security according to lattice estimator primal_bdd.
# Runtime 170 seconds (83ms/example).

# Num plaintext bits: 32, noise bits: 84
# Max representable value: 654624
context = hadal.create_context64(
    log_n=12,
    main_moduli=[288230376151760897, 288230376152137729],
    plaintext_modulus=4294991873,
    scaling_factor=3,
    seed="test_seed",
)
# 120 bits of security according to lattice estimator primal_bdd.
# Runtime 388 seconds (95ms/example).

key = hadal.create_key64(context)
rotation_key = hadal.create_rotation_key64(context, key)

# Prepare the dataset.
batch_size = context.num_slots.numpy()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)


@tf.function
def train_step(x, y, hidden_layer, output_layer):
    # Forward pass.
    y_1 = hidden_layer(x, training=True)
    y_pred = output_layer(y_1, training=True)

    # Backward pass.
    dJ_dy_pred = y.__rsub__(y_pred)  # Derivative of CCE loss and softmax.

    dJ_dw1, dJ_dx1 = output_layer.backward(dJ_dy_pred, rotation_key)

    # Mod reduce will reduce noise but increase the plaintext error.
    # if isinstance(dJ_dx1, hadal.ShellTensor64):
    #     dJ_dx1.get_mod_reduced()

    dJ_dw0, dJ_dx0_unused = hidden_layer.backward(dJ_dx1, rotation_key)

    # Only return the weight gradients at [0], not the bias gradients at [1].
    return dJ_dw1[0], dJ_dw0[0]


class TestMNISTBackprop(tf.test.TestCase):
    def _test_mnist_enc_backprop(self, use_fast_reduce_sum):
        # Create the layers
        hidden_layer = hadal_ml.ShellDense(
            64,
            activation=hadal_ml.relu,
            activation_deriv=hadal_ml.relu_deriv,
            is_first_layer=True,
            grad_reduction="fast" if use_fast_reduce_sum else "galois",
        )
        output_layer = hadal_ml.ShellDense(
            10,
            activation=tf.nn.softmax,
            # Do not set the derivative of the activation function for the last layer
            # in the model. The derivative of the categorical crossentropy loss function
            # times the derivative of a softmax is just y_pred - y (which is much easier
            # to compute than each of them individually). So instead just let the
            # loss function derivative incorporate y_pred - y and let the derivative
            # of this last layer's activation be a no-op.
            grad_reduction="fast" if use_fast_reduce_sum else "galois",
        )

        # Call the layers once to create the weights.
        y1 = hidden_layer(tf.zeros((batch_size, 784)))
        y2 = output_layer(y1)

        (x_batch, y_batch) = next(iter(train_dataset))

        # Plaintext backprop.
        output_layer_grad, hidden_layer_grad = train_step(
            x_batch, y_batch, hidden_layer, output_layer
        )

        # Encrypt y.
        enc_y_batch = hadal.to_encrypted(y_batch, key, context)

        # Encrypted Backprop.
        enc_output_layer_grad, enc_hidden_layer_grad = train_step(
            x_batch, enc_y_batch, hidden_layer, output_layer
        )

        # Decrypt the gradients.
        if use_fast_reduce_sum:
            fast_rotation_key = hadal.create_fast_rotation_key64(context, key)
            shell_output_layer_grad = hadal.to_tensorflow(
                enc_output_layer_grad, fast_rotation_key
            )
            shell_hidden_layer_grad = hadal.to_tensorflow(
                enc_hidden_layer_grad, fast_rotation_key
            )
        else:
            shell_output_layer_grad = hadal.to_tensorflow(enc_output_layer_grad, key)
            shell_hidden_layer_grad = hadal.to_tensorflow(enc_hidden_layer_grad, key)

        # Compare the gradients.
        self.assertAllClose(
            output_layer_grad,
            shell_output_layer_grad,
            atol=1 / context.scaling_factor * batch_size,
            rtol=1 / context.scaling_factor * 2,
        )

        self.assertAllClose(
            hidden_layer_grad,
            shell_hidden_layer_grad,
            atol=1 / context.scaling_factor * batch_size,
            rtol=1 / context.scaling_factor * 3,
        )

    def test_mnist_enc_backprop(self):
        for use_fast_reduce_sum in [False, True]:
            with self.subTest(
                f"{self._testMethodName} with use_fast_reduce_sum={use_fast_reduce_sum}."
            ):
                self._test_mnist_enc_backprop(use_fast_reduce_sum)


if __name__ == "__main__":
    unittest.main()
