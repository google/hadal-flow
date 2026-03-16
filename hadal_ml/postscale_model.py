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
import tensorflow as tf
import hadal_flow as hadal
from hadal_ml.private_base import PrivateBase


class PostScaleModel(PrivateBase):
    def __init__(
        self,
        *args,
        ubatch_per_batch=1,
        **kwargs,
    ):
        if not isinstance(ubatch_per_batch, int) or ubatch_per_batch <= 0:
            raise ValueError(
                "ubatch_per_batch must be a positive integer, got: "
                f"{ubatch_per_batch}"
            )
        self.ubatch_per_batch = ubatch_per_batch

        super().__init__(*args, **kwargs)

        for l in self.layers:
            if "hadal_ml" in getattr(l, "__module__", None):
                raise ValueError(
                    "hadal_ml.PostScaleSequential does not support hadal layers"
                )

    def call(self, inputs, training=False, with_softmax=True):
        prediction = super().call(inputs, training)
        if with_softmax:
            # Perform the last layer activation since it is removed for training
            # purposes.
            prediction = tf.nn.softmax(prediction)
        return prediction

    def _predict_and_jacobian(self, features, model):
        """
        Predicts the output for the given features and optionally computes the
        Jacobian.

        Args:
            features (tf.Tensor): Input features for the model.

        Returns:
            tuple: A tuple containing:
                - predictions (tf.Tensor): The model output after applying
                  softmax.
                - jacobians (list or tf.Tensor): The Jacobian of the last layer
                  preactivation with respect to the model weights.
        """
        # Make sure model is a tf.keras.Model
        if not isinstance(model, tf.keras.Model):
            raise ValueError("model must be a tf.keras.Model")

        with tf.GradientTape(
            persistent=tf.executing_eagerly() or self.jacobian_pfor
        ) as tape:
            predictions = model.call(features, training=True)

        jacobians = tape.jacobian(
            predictions,
            self.trainable_variables,
            # unconnected_gradients=tf.UnconnectedGradients.ZERO, broken with pfor
            parallel_iterations=self.jacobian_pfor_iterations,
            experimental_use_pfor=self.jacobian_pfor,
        )
        # ^  layers list x shape: [batch size, num output classes, weights]

        # Compute the last layer's activation manually since we skipped it above.
        predictions = tf.nn.softmax(predictions)

        return predictions, jacobians

    def _backward(self, dJ_dz, jacobians):
        # Scale each gradient. Since 'scalars' may be a vector of
        # ciphertexts, this requires multiplying plaintext gradient for the
        # specific layer (2d) by the ciphertext (scalar).
        grads = []
        for j in jacobians:
            # Ignore the batch size and num output classes dimensions and
            # recover just the number of dimensions in the weights.
            num_weight_dims = len(j.shape) - 2

            # Make the scalars the same shape as the gradients so the
            # multiplication can be broadcasted.
            dJ_dz_exp = dJ_dz
            for _ in range(num_weight_dims):
                dJ_dz_exp = hadal.expand_dims(dJ_dz_exp, axis=-1)

            # Scale the jacobian.
            scaled_grad = dJ_dz_exp * j
            # ^ shape: [batch_size, num output classes, weights]

            # Sum over the output classes. At this point, this is a gradient
            # and no longer a jacobian.
            scaled_grad = hadal.reduce_sum(scaled_grad, axis=1)
            # ^ shape: [batch_size, weights]

            grads.append(scaled_grad)

        return grads

    def compute_postscale_precursors_ubatch(
        self, ubatch_features, model, scaling_factor
    ):
        prediction, jacobians = self._predict_and_jacobian(ubatch_features, model)

        # Perform PostScale (dJ/dz * dz/dw) in plaintext for every
        # possible label using the worst casse quantization of the
        # jacobian.
        with tf.name_scope("sensitivity_analysis"):
            per_class_norms = tf.TensorArray(
                dtype=tf.keras.backend.floatx(),
                size=self.out_classes,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([ubatch_features.shape[0]]),
            )
            worst_case_jacobians = [
                hadal.largest_case_rounding(j, scaling_factor) for j in jacobians
            ]
            worst_case_prediction = hadal.largest_case_rounding(
                prediction, scaling_factor
            )

            def cond(possible_label_i, _):
                return possible_label_i < self.out_classes

            def body(possible_label_i, per_class_norms):
                possible_label = tf.one_hot(
                    possible_label_i,
                    self.out_classes,
                    dtype=tf.keras.backend.floatx(),
                )
                dJ_dz = worst_case_prediction - possible_label
                possible_grads = self._backward(dJ_dz, worst_case_jacobians)
                possible_norms = tf.vectorized_map(self.gradient_norms, possible_grads)

                per_class_norms = per_class_norms.write(
                    possible_label_i, possible_norms
                )
                return possible_label_i + 1, per_class_norms

            # Using a tf.while_loop (vs. a python for loop) is preferred
            # as it does not encode the unrolled loop into the graph and
            # also allows explicit control over the loop's parallelism.
            # Increasing parallel_iterations may be faster at the expense
            # of memory usage.
            possible_label_i = tf.constant(0)
            _, per_class_norms = tf.while_loop(
                cond,
                body,
                [possible_label_i, per_class_norms],
                parallel_iterations=1,
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorSpec(None, dtype=tf.variant),
                ],
            )

            per_class_per_example_norms = per_class_norms.stack()
            # ^ shape: [num output classes, batch_size]
            per_example_per_class_norms = tf.transpose(per_class_per_example_norms)
            # ^ shape: [batch_size, num output classes]
            return prediction, jacobians, per_example_per_class_norms

    def compute_grads(self, features, enc_labels):
        scaling_factor = (
            enc_labels.scaling_factor
            if hasattr(enc_labels, "scaling_factor")
            else float("inf")
        )
        scaling_factor = tf.cast(scaling_factor, dtype=tf.keras.backend.floatx())

        with tf.device(self.features_party_dev):
            num_devices = len(self.jacobian_devices)
            batch_size = features.shape[0]
            if batch_size % num_devices != 0:
                raise ValueError("Batch size must be divisible by number of devices.")
            batch_size_per_device = batch_size // num_devices
            if batch_size_per_device % self.ubatch_per_batch != 0:
                raise ValueError(
                    "Batch size (per device) must be divisible by ubatch_per_batch."
                )
            ubatch_size = batch_size_per_device // self.ubatch_per_batch
            all_devs_ubatch_size = batch_size // self.ubatch_per_batch
            num_ubatches = num_devices * self.ubatch_per_batch

            # Due to TensorFlow errata, reshaping this tensor only works when
            # indexed first by the ubatch dimension, then by the device
            # dimension. Without this ordering, the data is silently corrupted
            # resulting in poor accuracy models after training. It is only
            # observed when both devices and ubatch_per_batch > 1.
            split_features = tf.reshape(
                features,
                [self.ubatch_per_batch, num_devices, ubatch_size] + features.shape[1:],
            )

        # Iterate over the ubatch dimension first, then parallelize over the
        # available devices. This ensures when the features are large, they are
        # only copied to the device in small chunks (of size ubatch).
        with tf.device(self.features_party_dev):
            ubatch = tf.constant(0)
            predictions_ta_dev = tf.TensorArray(
                dtype=tf.keras.backend.floatx(),
                size=self.ubatch_per_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=[all_devs_ubatch_size, self.out_classes],
            )
            jacobians_tas_dev = [
                tf.TensorArray(
                    dtype=v.dtype,
                    size=self.ubatch_per_batch,
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=[all_devs_ubatch_size, self.out_classes] + v.shape,
                )
                for v in self.trainable_variables
            ]
            jacobians_norms_ta_dev = tf.TensorArray(
                dtype=tf.keras.backend.floatx(),
                size=self.ubatch_per_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=[all_devs_ubatch_size, self.out_classes],
            )

            def cond(ubatch, *args):
                return ubatch < self.ubatch_per_batch

            def body(ubatch, predictions_ta, jacobians_tas, jacobians_norms_ta):
                # Iterate over the devices and compute the results
                device_predictions = []
                device_jacobians = [[] for _ in self.trainable_variables]
                device_jacobians_norms = []
                for dev_i, (dev, dev_model) in enumerate(
                    zip(self.jacobian_devices, self.jacobian_shadow_models)
                ):
                    dev_features = split_features[ubatch][dev_i]
                    with tf.device(dev):
                        dev_preds, dev_jacs, dev_jac_norms = (
                            self.compute_postscale_precursors_ubatch(
                                tf.identity(dev_features), dev_model, scaling_factor
                            )
                        )
                        device_predictions.append(dev_preds)
                        for j, j_list in enumerate(device_jacobians):
                            j_list.append(dev_jacs[j])
                        device_jacobians_norms.append(dev_jac_norms)

                # Concat the results from each device
                ubatch_predictions = tf.concat(device_predictions, axis=0)
                ubatch_jacobians = [
                    tf.concat(j_list, axis=0) for j_list in device_jacobians
                ]
                ubatch_jacobians_norms = tf.concat(device_jacobians_norms, axis=0)

                # Store the results in the TensorArrays.
                with tf.device(self.features_party_dev):
                    predictions_ta = predictions_ta.write(ubatch, ubatch_predictions)
                    for j, jac in enumerate(ubatch_jacobians):
                        jacobians_tas[j] = jacobians_tas[j].write(ubatch, jac)
                    jacobians_norms_ta = jacobians_norms_ta.write(
                        ubatch, ubatch_jacobians_norms
                    )

                    return (
                        ubatch + 1,
                        predictions_ta,
                        jacobians_tas,
                        jacobians_norms_ta,
                    )

            (
                _,
                predictions_ta_dev,
                jacobians_tas_dev,
                jacobians_norms_ta_dev,
            ) = tf.while_loop(
                cond,
                body,
                [
                    ubatch,
                    predictions_ta_dev,
                    jacobians_tas_dev,
                    jacobians_norms_ta_dev,
                ],
                parallel_iterations=1,  # only 1 ubatch at a time to save memory
            )

            predictions = predictions_ta_dev.concat()
            jacobians = [layer.concat() for layer in jacobians_tas_dev]
            jacobians_norms = jacobians_norms_ta_dev.concat()

        with tf.device(self.features_party_dev):
            per_example_norms = tf.reduce_max(jacobians_norms, axis=1)

            if type(enc_labels) is tf.Tensor:
                tf.debugging.assert_equal(
                    tf.shape(predictions),
                    tf.shape(enc_labels),
                    message="Predictions and labels must have the same shape.",
                )
            # The base class ensures that when the loss is CCE, the last
            # layer's activation is softmax. The derivative of these two
            # functions is simple subtraction.
            dJ_dz = enc_labels.__rsub__(predictions)
            # ^ shape: [batch_size, num output classes]

            grads = self._backward(dJ_dz, jacobians)

        return grads, per_example_norms, predictions
