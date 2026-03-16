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

epochs = 2
batch_size = 2**12

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = np.reshape(x_train, (-1, 784)), np.reshape(x_test, (-1, 784))
x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)


# Create the layers
hidden_layer = hadal_ml.ShellDense(
    64,
    activation=hadal_ml.relu,
    activation_deriv=hadal_ml.relu_deriv,
    use_bias=True,
    is_first_layer=True,
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
    use_bias=True,
)

# Call the layers once to create the weights.
y1 = hidden_layer(tf.zeros((batch_size, 784)))
y2 = output_layer(y1)

optimizer = tf.keras.optimizers.Adam(0.01)


@tf.function
def train_step(x, y):
    # Forward pass.
    y_1 = hidden_layer(x, training=True)
    y_pred = output_layer(y_1, training=True)

    # Backward pass.
    dJ_dy_pred = y.__rsub__(y_pred)  # Derivative of CCE loss and softmax.

    dJ_dw1, dJ_dx1 = output_layer.backward(dJ_dy_pred)

    dJ_dw0, dJ_dx0_unused = hidden_layer.backward(dJ_dx1)

    return dJ_dw1, dJ_dw0


class TestMNISTBackprop(tf.test.TestCase):

    # Test plaintext training using hadal_ml primitives.
    def test_mnist_plaintext_backprop(self):

        # Test both eager and graph mode.
        for is_eager in [True, False]:
            tf.config.run_functions_eagerly(is_eager)

            # Train the model.
            for epoch in range(epochs):
                for step, (x_batch, y_batch) in enumerate(
                    train_dataset.take(batch_size)
                ):
                    # Plaintext backprop splitting the batch in half vertically.
                    output_layer_grad, hidden_layer_grad = train_step(x_batch, y_batch)

                    # Reduce sum the gradients.
                    output_layer_grad = [
                        tf.reduce_sum(g, axis=0) for g in output_layer_grad
                    ]
                    hidden_layer_grad = [
                        tf.reduce_sum(g, axis=0) for g in hidden_layer_grad
                    ]

                    # To directly apply the weights, use the following:
                    # output_layer.weights[0] = output_layer.weights[0] - 0.01 * output_layer_grad[0]
                    # hidden_layer.weights[0] = hidden_layer.weights[0] - 0.01 * hidden_layer_grad[0]

                    optimizer.apply_gradients(
                        zip(
                            output_layer_grad + hidden_layer_grad,
                            output_layer.weights + hidden_layer.weights,
                        )
                    )

                average_accuracy = 0.0
                for x, y in val_dataset:
                    y_pred = output_layer(hidden_layer(x))
                    accuracy = tf.reduce_mean(
                        tf.cast(
                            tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1)),
                            tf.float32,
                        )
                    )
                    average_accuracy += accuracy
                average_accuracy /= len(val_dataset)

                print(f"Accuracy: {average_accuracy}")

            # Ensure the model is learning.
            self.assertAllGreater(average_accuracy, 0.9)


if __name__ == "__main__":
    unittest.main()
