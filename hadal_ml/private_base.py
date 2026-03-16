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
import tensorflow.keras as keras
from tensorflow.keras.callbacks import CallbackList
import hadal_flow as hadal
import time
import gc
import hadal_ml


class PrivateBase(keras.Model):

    def __init__(
        self,
        inputs=None,
        outputs=None,
        backprop_context_fn=None,
        noise_context_fn=None,
        noise_multiplier_fn=lambda batch_size: 1.0,
        labels_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        features_party_dev="/job:localhost/replica:0/task:0/device:CPU:0",
        noise_max_scale=5.0e9,
        noise_base_scale=7.6,
        cache_path=None,
        jacobian_pfor=False,
        jacobian_pfor_iterations=None,
        jacobian_devices=None,
        disable_he_backprop_INSECURE=False,
        disable_masking_INSECURE=False,
        simple_noise_INSECURE=False,
        clip_threshold=1.0,
        check_overflow_INSECURE=False,
        *args,
        **kwargs,
    ):
        super().__init__(inputs=inputs, outputs=outputs, *args, **kwargs)
        if backprop_context_fn is None:
            raise ValueError(
                "backprop_context_fn must be provided. It is used to set "
                "encryption parameters for backpropagation."
            )
        self.backprop_context_fn = backprop_context_fn
        if noise_context_fn is None:
            raise ValueError(
                "noise_context_fn must be provided. It is used to set "
                "encryption parameters for DP noise protocol."
            )
        self.noise_context_fn = noise_context_fn
        self.noise_multiplier_fn = noise_multiplier_fn
        self.labels_party_dev = labels_party_dev
        self.features_party_dev = features_party_dev
        self.cache_path = cache_path
        self.jacobian_pfor = jacobian_pfor
        self.jacobian_pfor_iterations = jacobian_pfor_iterations
        self.jacobian_devices = (
            [features_party_dev] if jacobian_devices is None else jacobian_devices
        )
        self.disable_he_backprop_INSECURE = disable_he_backprop_INSECURE
        self.disable_masking_INSECURE = disable_masking_INSECURE
        self.simple_noise_INSECURE = simple_noise_INSECURE
        self.check_overflow_INSECURE = check_overflow_INSECURE
        self.clip_threshold = clip_threshold

        self.dataset_prepped = False
        self.uses_cce_and_softmax = False

        self.dg_params = hadal.DiscreteGaussianParams(
            max_scale=noise_max_scale, base_scale=noise_base_scale
        )

        if self.disable_he_backprop_INSECURE and self.jacobian_pfor:
            print(
                "WARNING: `jacobian_pfor` may be incompatible with `disable_he_backprop_INSECURE`."
            )

        if len(self.layers) == 0:
            raise ValueError("The model must have at least one layer.")

        self.out_classes = self.outputs[0].shape[-1]

        if len(self.jacobian_devices) == 0:
            raise ValueError("No devices specified for Jacobian computation.")

        if not self.disable_masking_INSECURE:
            if self.disable_he_backprop_INSECURE:
                raise ValueError(
                    "The masking subprotocol requires HE backpropagation to be enabled."
                )

        if self.simple_noise_INSECURE:
            # Simple noise clipping (like in traditional DP-SGD) requires
            # knowing the gradients in plaintext, so encryption and masking must
            # be disabled.
            if not (
                self.disable_he_backprop_INSECURE
                and self.disable_masking_INSECURE
                and self.clip_threshold > 0.0
            ):
                raise ValueError(
                    "Simple gradient clipping (like traditional DP-SGD) requires "
                    "encryption, masking, and noise protocols are disabled. "
                    "The clipping threshold must be > 0. Saw: "
                    f"disable_he_backprop_INSECURE={self.disable_he_backprop_INSECURE}, "
                    f"disable_masking_INSECURE={self.disable_masking_INSECURE}, "
                    f"clip_threshold={self.clip_threshold}"
                )

        # Create a shadow model on each device for computing the Jacobian.
        self.jacobian_shadow_models = []
        for dev in self.jacobian_devices:
            with tf.device(dev):
                dev_model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
                self.jacobian_shadow_models.append(dev_model)

    def compile(self, loss, **kwargs):
        if isinstance(loss, tf.keras.losses.CategoricalCrossentropy):
            if not (
                self.layers[-1].activation is tf.keras.activations.softmax
                or self.layers[-1].activation is tf.nn.softmax
            ):
                raise ValueError(
                    "When using CCE loss, the model must use a softmax activation function on the final layer. Instead saw",
                    self.layers[-1].activation,
                    self.layers,
                )

            # Unset the last layer's activation function. Derived classes handle
            # the softmax activation + cce manually.
            self.layers[-1].activation = keras.activations.linear
            self.layers[-1].activation_deriv = None
            self.uses_cce_and_softmax = True

        else:
            print(
                "WARNING: Not using CCE loss function. The DP noise sensitivity will not automatically include the gradient of the loss function and must be taken into account manually in the noise_multiplier_fn callback."
            )

        self.layers[0].is_first_layer = True

        super().compile(
            jit_compile=False,  # Disable XLA, no CPU op for hadal_ml's TensorArrayV2.
            loss=loss,
            **kwargs,
        )
        # Build the optimizer to create any internal variables in this scope.
        self.optimizer.build(self.trainable_variables)

        # Build the shadow models.
        for dev, dev_model in zip(self.jacobian_devices, self.jacobian_shadow_models):
            with tf.device(dev):
                dev_model.build(self.inputs[0].shape)
                dev_model.compile(loss=loss, **kwargs)

    def train_step(self, features, labels):
        """
        Executes a single training step.

        Args:
            features: The input features for the training step.
            labels: The corresponding labels for the training step.

        Returns:
            metrics (dict): The metrics resulting from the training step.
        """
        # When called from outside this class, the caches for encryption keys
        # and contexts are not guaranteed to be populated, so set
        # read_key_from_cache to False.
        metrics, batch_size_should_be = self.shell_train_step(
            features, labels, read_key_from_cache=False, apply_gradients=True
        )
        return metrics

    @tf.function
    def train_step_tf_func(
        self, features, labels, read_key_from_cache, apply_gradients
    ):
        """
        tf.function wrapper around shell train step.
        """
        return self.shell_train_step(
            features, labels, read_key_from_cache, apply_gradients
        )

    def compute_grads(self, features, enc_labels):
        raise NotImplementedError()  # Should be overloaded by the subclass.

    def batch_size_from_trace(self, train_features, train_labels):
        """
        Determine the batch size from the graph.
        """
        # Call the training step with keygen to trace the graph. Note, since the
        # graph is not executed, the caches for encryption keys and contexts are
        # not written to disk.
        func = self.train_step_tf_func.get_concrete_function(
            next(iter(train_features)),
            next(iter(train_labels)),
            read_key_from_cache=False,
            apply_gradients=False,
        )

        # Optimize the graph using hadal HE-specific optimizers.
        optimized_func = hadal.optimize_shell_graph(
            func, skip_convert_to_constants=True
        )
        optimized_graph = optimized_func.graph

        def find_node_by_op(g, name):
            for node in g.as_graph_def().node:
                if node.op == name:
                    return node
            raise ValueError(f"Node {name} not found in graph.")

        # Using parameters in the optimized graph, create the context and
        # keys for use during training. The parameters are pulled from the
        # graph because if autocontext is used, these parameters are not
        # known until the graph optimization pass is finished.
        context_node = find_node_by_op(optimized_graph, "ContextImport64")

        def get_tensor_by_name(g, name):
            for node in g.as_graph_def().node:
                if node.name == name:
                    return tf.make_ndarray(node.attr["value"].tensor)
            raise ValueError(f"Node {name} not found in graph.")

        log_n = get_tensor_by_name(optimized_graph, context_node.input[0]).tolist()
        batch_size = 2**log_n
        return batch_size

    def prep_dataset_for_model(self, train_features, train_labels, fast=True):
        """
        Prepare dataset for training with HE, setting the batch size to
        the encryption ring degree.

        When `fast` is False, run the training loop once on dummy inputs to
        determine the batch size. Note the batch size is not always known
        during graph construction when using autocontext, only after graph
        optimization.

        When `fast` is True, trace the graph (without executing it) to determine
        the batch size.

        Args:
            train_features (tf.data.Dataset): The training features dataset.
            train_labels (tf.data.Dataset): The training labels dataset.
            fast (bool, optional): Whether to use the fast method to determine
                the batch size. Defaults to True.

        Returns:
            tuple: A tuple containing the re-batched training features and labels.

        """
        if self.disable_he_backprop_INSECURE and self.simple_noise_INSECURE:
            # If neither encryption nor noise are enabled, no encryption is used
            # and the batch size is the same as the input batch size.
            self.batch_size = next(iter(train_features)).shape[0]
        elif not fast:
            # Run the training loop once on dummy data. This writes encryption
            # contexts and keys to caches.
            metrics, batch_size_should_be = self.train_step_tf_func(
                next(iter(train_features)),
                next(iter(train_labels)),
                read_key_from_cache=False,
                apply_gradients=False,
            )
            self.batch_size = batch_size_should_be.numpy()
        else:
            self.batch_size = self.batch_size_from_trace(train_features, train_labels)

        with tf.device(self.features_party_dev):
            train_features = train_features.rebatch(
                self.batch_size, drop_remainder=True
            )
        with tf.device(self.labels_party_dev):
            train_labels = train_labels.rebatch(self.batch_size, drop_remainder=True)

        self.dataset_prepped = True
        return train_features, train_labels

    def fit(
        self,
        features_dataset,
        labels_dataset,
        epochs=1,
        batch_size=32,
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        steps_per_epoch=None,
        verbose=1,
        initial_epoch=0,
    ):
        """
        Train the model.

        This custom training loop supports inputs from multiple datasets
        (features and laabels), each of which can be located on a different
        device.

        Args:
            features_dataset (tf.data.Dataset): The dataset containing the
                features for training.
            labels_dataset (tf.data.Dataset): The dataset containing the labels
                for training.
            epochs (int, optional): Number of epochs to train the model.
                Defaults to 1.
            batch_size (int, optional): Number of samples per gradient update.
                Defaults to 32.
            callbacks (list, optional): List of `keras.callbacks.Callback`
                instances. Defaults to None.
            validation_data (tuple, optional): Data on which to evaluate the
                loss and any model metrics at the end of each epoch. Defaults to
                None.
            steps_per_epoch (int, optional): Total number of steps (batches of
                samples) before declaring one epoch finished and starting the
                next epoch. Defaults to None.
            verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress
                bar. Defaults to 1.

        Returns:
            History: A keras `History` object. Its `History.history` attribute
                is a record of training loss values and metrics values at
                successive epochs, as well as validation loss values and
                validation metrics values (if applicable).
        """
        # Prevent TensorFlow from placing ops on devices which were not
        # explicitly assigned for security reasons.
        tf.config.set_soft_device_placement(False)

        # Turn on the shell optimizers.
        hadal.enable_optimization()

        # Enable randomized rounding.
        hadal.enable_randomized_rounding()

        # Make sure the noise multiplier is set to something before calling
        # prep_dataset_for_model(). It can be any positive value, since it will
        # be overwritten later once the batch size is known.
        self.noise_multiplier = 1.0

        if not self.dataset_prepped:
            features_dataset, labels_dataset = self.prep_dataset_for_model(
                features_dataset, labels_dataset
            )
            tf.keras.backend.clear_session()
            gc.collect()

        # Now that the batch size is known, set the noise multiplier and make
        # sure it is valid.
        self.noise_multiplier = self.noise_multiplier_fn(self.batch_size)

        if self.noise_multiplier is None:
            raise ValueError(
                "The noise multiplier function must return a value. Saw",
                self.noise_multiplier,
            )

        if self.noise_multiplier == 0.0:
            self.simple_noise_INSECURE = True

        # Calculate samples if possible.
        if steps_per_epoch is None:
            samples = None
        else:
            samples = steps_per_epoch * self.batch_size

        # Initialize callbacks.
        callback_list = CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            batch_size=self.batch_size,
            epochs=epochs,
            steps=steps_per_epoch,
            samples=samples,
            verbose=verbose,
            do_validation=validation_data is not None,
            metrics=list(self.metrics_names),
        )

        # Begin training.
        callback_list.on_train_begin()
        logs = {}
        subsequent_run = False

        for epoch in range(initial_epoch, epochs):
            if self.stop_training:
                break
            callback_list.on_epoch_begin(epoch, logs)
            start_time = time.time()
            self.reset_metrics()

            # Training loop.
            for step, (batch_x, batch_y) in enumerate(
                zip(features_dataset, labels_dataset)
            ):
                callback_list.on_train_batch_begin(step, logs)
                # The caches for encryption keys and contexts have already been
                # populated during the dataset preparation step. Set
                # read_key_from_cache to True.
                logs, batch_size_should_be = self.train_step_tf_func(
                    batch_x,
                    batch_y,
                    read_key_from_cache=subsequent_run,
                    apply_gradients=True,
                )
                subsequent_run = True
                callback_list.on_train_batch_end(step, logs)
                gc.collect()
                if steps_per_epoch is not None and step + 1 >= steps_per_epoch:
                    break

            # Validation loop.
            if validation_data is not None:
                # Reset metrics
                self.reset_metrics()

                for steps, (val_x_batch, val_y_batch) in enumerate(validation_data):
                    val_y_pred = self(
                        val_x_batch,
                        training=False,
                        with_softmax=self.uses_cce_and_softmax,
                    )
                    # Update validation metrics
                    for m in self.metrics:
                        if m.name == "loss":
                            loss = self.compute_loss(y=val_y_batch, y_pred=val_y_pred)
                            m.update_state(loss)
                        else:
                            m.update_state(val_y_batch, val_y_pred)

                    if validation_steps is not None and steps + 1 >= validation_steps:
                        break
                metric_results = {m.name: m.result() for m in self.metrics}

                # TensorFlow 2.18.0 added a "CompiledMetrics" metric which holds
                # metrics passed to compile in it's own dictionary. Keras wants
                # all metrics to be returned as a flat dictionary. Here we
                # flatten the dictionary.
                result = {}
                for key, value in metric_results.items():
                    if isinstance(value, dict):
                        result.update(value)  # add subdict directly into the dict
                    else:
                        result[key] = value  # non-subdict elements are just copied

                logs.update({f"val_{name}": result for name, result in result.items()})

            # End of epoch.
            logs["time"] = time.time() - start_time

            # Update the steps in callback parameters with actual steps completed
            if steps_per_epoch is None:
                steps_per_epoch = step + 1
                samples = steps_per_epoch * self.batch_size
                callback_list.params["steps"] = steps_per_epoch
                callback_list.params["samples"] = samples
            callback_list.on_epoch_end(epoch, logs)

        # End of training.
        callback_list.on_train_end(logs)
        return self.history

    def flatten_and_pad_grad(self, grad, dim_0_sz):
        num_el = tf.cast(tf.size(grad), dtype=tf.int64)

        flat = tf.reshape(grad, [-1])
        pad_len = dim_0_sz - tf.math.floormod(num_el, dim_0_sz)
        padded = tf.concat([flat, tf.zeros(pad_len, dtype=grad.dtype)], axis=0)
        flat_padded = tf.reshape(padded, [dim_0_sz, -1])
        metadata = {
            "original_shape": tf.shape(grad),
            "padding": pad_len,
        }
        return flat_padded, metadata

    def unflatten_grad(self, grad, metadata):
        padding = metadata["padding"]
        orig_shape = metadata["original_shape"]

        flat_grad = tf.reshape(grad, [-1])
        unpadded_flat_grad = flat_grad[:-padding]
        shaped_grad = tf.reshape(unpadded_flat_grad, orig_shape)
        return shaped_grad

    def flatten_and_pad_grad_list(self, grads_list, dim_0_sz):
        all_grads = []
        all_metadata = []
        for grad in grads_list:
            flat, metadata = self.flatten_and_pad_grad(grad, dim_0_sz)
            all_grads.append(flat)
            all_metadata.append(metadata)

        return all_grads, all_metadata

    def unflatten_grad_list(self, flat_grads, metadata_list):
        grads = []
        for flat, metadata in zip(flat_grads, metadata_list):
            grad = self.unflatten_grad(flat, metadata)
            grads.append(grad)

        return grads

    def _expand_dims_like(self, e, v):
        new_shape = tf.concat(
            [
                tf.shape(v)[0:1],
                tf.ones_like(tf.shape(v), dtype=tf.int32)[:-1],
            ],
            axis=0,
        )
        return tf.reshape(e, new_shape)

    def spoof_int_gradients(self, grads):
        """
        Spoofs the gradients to have dtype int64 and a scaling factor of 1.

        Args:
            grads (list of hadal.ShellTensor): The list of gradient tensors
                to be spoofed.

        Returns:
            list of hadal.ShellTensor64: The list of gradient tensors with
                dtype int64 and scaling factor 1.
        """
        int64_grads = [
            hadal.ShellTensor64(
                _raw_tensor=g._raw_tensor,
                _context=g._context,
                _level=g._level,
                _num_mod_reductions=g._num_mod_reductions,
                _underlying_dtype=tf.int64,  # Spoof the dtype.
                _scaling_factor=float(1),  # Spoof the scaling factor.
                _is_enc=g._is_enc,
                _is_fast_rotated=g._is_fast_rotated,
            )
            for g in grads
        ]
        return int64_grads

    def unspoof_int_gradients(self, int64_grads, orig_scaling_factors):
        """
        Recover the floating point gradients from spoofed int64 gradients using
        the original scaling factors.

        Args:
            int64_grads (list of tf.Tensor): List of gradients that have been
                spoofed as int64 with scaling factor 1.
            orig_scaling_factors (list of float): List of original scaling
                factors used to spoof the gradients.

        Returns:
            list of tf.Tensor: List of recovered floating point gradients.
        """
        grads = [
            tf.cast(g, tf.keras.backend.floatx()) / s
            for g, s in zip(int64_grads, orig_scaling_factors)
        ]
        return grads

    def mask_gradients(self, context, grads):
        """
        Adds random masks to the encrypted gradients within the range [-t/2,
        t/2].

        Args:
            context (object): The context containing the plaintext modulus.
            grads (list): A list of gradient tensors to be masked.

        Returns:
        tuple: A tuple containing:
            - int64_grads (list): The masked gradients with spoofed dtype and
              scaling factor.
            - masks (list): The random masks added to the gradients.
            - mask_scaling_factors (list): The original scaling factors of the
              gradients.
        """
        t = tf.cast(tf.identity(context.plaintext_modulus), dtype=tf.int64)
        t_half = t // 2

        masks = [
            tf.random.uniform(
                hadal.shape(g),
                dtype=tf.int64,
                minval=-t_half,
                maxval=t_half,
            )
            for g in grads
        ]

        # Convert to shell plaintexts manually to avoid using the scaling factor
        # in the 'grads[i]' contexts when the addition happens in the next line
        # below.
        shell_masks = [
            hadal.to_shell_plaintext(m, context, override_scaling_factor=1)
            for m in masks
        ]

        # Add the masks.
        masked_grads = [(g + m) for g, m in zip(grads, shell_masks)]

        return masked_grads, masks

    def unmask_gradients(self, context, grads, masks):
        """
        Unmasks the gradients by subtracting the masks, and converting to
        floating point representation using the scaling factors.

        Args:
            context: The context in which the operation is performed.
            grads (list of tf.Tensor): The gradients to be unmasked.
            masks (list of tf.Tensor): The masks to be subtracted from the
              gradients.
            mask_scaling_factors (list of float): The scaling factors for the
              masks.

        Returns:
            list of tf.Tensor: The unmasked gradients.
        """
        # Sum the masks over the batch.
        sum_masks = [hadal.reduce_sum_with_mod(m, 0, context, 1) for m in masks]

        # Unmask the batch gradient.
        masks_and_grads = [tf.stack([-m, g]) for m, g in zip(sum_masks, grads)]
        unmasked_grads = [
            hadal.reduce_sum_with_mod(mg, 0, context, 1) for mg in masks_and_grads
        ]

        return unmasked_grads

    def compute_noise_factors(
        self, context, secret_key, max_two_norm, mask_scaling_factors
    ):
        """
        Computes noise selection vectors, as part of the distributed noise
        sampling protocol to sample the discrete gaussian distribution.

        Args:
            context: An object containing the context information, including the
              number of slots for encryption.
            secret_key: The secret key used for encryption.
            max_two_norm: Maximum L2 norm of per-example gradients.
              factors.

        Returns:
            dict: A dictionary containing the following arrays, one per scaling factor:
                - enc_a: The encrypted noise selection vector `a`.
                - enc_b: The encrypted noise selection vector `b`.

        Raises:
            tf.errors.InvalidArgumentError: If the bounded sensitivity exceeds
              the maximum noise scale defined in `self.dg_params`.
        """
        sensitivity = tf.cast(max_two_norm, dtype=tf.float32)

        # This code may run before the batch size is known, thus the noise
        # multiplier is not yet set. For instance, the graph is traced
        # (including this function) to determine the batch size, which is then
        # used to compute the noise multiplier.
        if hasattr(self, "noise_multiplier"):
            sensitivity *= self.noise_multiplier

        noise_factors = {
            "enc_as": [],
            "enc_bs": [],
        }

        for sf in mask_scaling_factors:
            # When the noise is added to the gradients, the gradients are scaled
            # by the `mask scaling factors`. The noise must be scaled by the
            # same amount.
            scaled_sensitivity = sensitivity * sf

            # Ensure the sensitivity is at least as large as the base scale, since
            # the base_scale is the smallest scale distribution which the protocol
            # can sample.
            bounded_sensitivity = tf.maximum(
                scaled_sensitivity, self.dg_params.base_scale
            )

            # The noise protocol generates samples in the range [-6s, 6s], where
            # s is the scale. Ensure the maximum noise is less than the
            # plaintext modulus.
            tf.assert_less(
                bounded_sensitivity * 6,
                tf.cast(context.plaintext_modulus, tf.float32),
                message="Sensitivity is too large for the noise context's plaintext modulus and WILL overflow. Reduce the sensitivity by reducing the gradient norms (e.g. reducing the backprop context's scaling factor), or increase the noise context's plaintext modulus.",
            )

            tf.assert_less(
                bounded_sensitivity,
                self.dg_params.max_scale,
                message="Sensitivity is too large for the maximum noise scale.",
            )

            a, b = hadal.sample_centered_gaussian_f(bounded_sensitivity, self.dg_params)

            def _prep_noise_factor(x):
                x = tf.expand_dims(x, 0)
                x = tf.expand_dims(x, 0)
                x = tf.repeat(x, context.num_slots, axis=0)
                return hadal.to_encrypted(x, secret_key, context)

            noise_factors["enc_as"].append(_prep_noise_factor(a))
            noise_factors["enc_bs"].append(_prep_noise_factor(b))

        return noise_factors

    def noise_gradients(self, context, flat_grads, noise_factors):
        """
        Adds encrypted noise to the gradients for differential privacy, as
        part of the distributed noise sampling protocol.

        Args:
            context: The context in which the noise is sampled.
            flat_grads: The flattened gradients to which noise is added.
            enc_a: Encrypted noise selection vector.
            enc_b: Encrypted noise selection vector.

        Returns:
            Tensor: The noised gradients.
        """

        noised_grads = []
        for grad, enc_a, enc_b in zip(
            flat_grads, noise_factors["enc_as"], noise_factors["enc_bs"]
        ):

            def _sample_noise():
                n = hadal.sample_centered_gaussian_l(
                    context,
                    tf.size(grad, out_type=tf.int64),
                    self.dg_params,
                )
                # The shape prefix of the noise samples must match the shape
                # of the masked gradients. The last dimension is the noise
                # sub-samples.
                n = tf.reshape(n, tf.concat([tf.shape(grad), [tf.shape(n)[1]]], axis=0))
                return n

            y1 = _sample_noise()
            y2 = _sample_noise()

            enc_x1 = hadal.reduce_sum(enc_a * y1, axis=2)
            enc_x2 = hadal.reduce_sum(enc_b * y2, axis=2)
            enc_noise = enc_x1 + enc_x2
            noised_grads.append(enc_noise + grad)

        return noised_grads

    def clip_gradient(self, grads_list):
        """
        Clips a single example gradient to have L2 norm of `self.clip_threshold`.

        Args:
            grads_list (list of tf.Tensor): List of gradient tensors for a
              single example to be clipped.
        Returns:
            list of tf.Tensor: List of clipped gradient tensors.
        """
        return tf.clip_by_global_norm(grads_list, self.clip_threshold)[0]

    def gradient_norms(self, grads_list):
        """
        A helper function to compute the global norm of a list of gradients that
        is safe to run on GPU.

        It is most convenient to use tf.linalg.global_norm to compute the global
        norm of a list of tensors. However, it causes the following error when
        called on GPU

        WARNING:tensorflow:Using a while_loop for converting L2Loss cause there is no registered converter for this op.
        """
        # The simplest method which should work:
        # return tf.linalg.global_norm(grads_list)

        # Explicitly set the shape of the returned norm to avoid issues
        # with shape inference later in the graph. This also should work.
        # n = tf.linalg.global_norm(grads_list)
        # n.set_shape([])
        # return n

        grads_flat = tf.nest.flatten(grads_list)
        squared_l2_norms = [
            tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
        ]
        global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
        return global_norm
        div = tf.maximum(global_norm / self.clip_threshold, 1.0)
        return div

    def warn_on_overflow(self, grads, scaling_factors, plaintext_modulus, message):
        """
        Print a warning if any of the gradients indicate overflow.

        This function checks if any of the gradients are within the ranges
        [-t/2, -t/4] or [t/4, t/2], where t is the plaintext modulus, which may
        indicate an overflow. The check takes into account the scaling factors,

        Args:
            grads (list of tf.Tensor): List of gradient tensors.
            scaling_factors (list of float): List of scaling factors
              corresponding to each gradient tensor.
            plaintext_modulus (float): The plaintext modulus value.
            message (str): The message to print if an overflow is detected.
        """
        # t = tf.cast(plaintext_modulus, grads[0].dtype)
        t_half = plaintext_modulus / 2

        over_by = [
            tf.reduce_max(tf.abs(g) - tf.cast(t_half / 2 / s, g.dtype))
            for g, s in zip(grads, scaling_factors)
        ]
        max_over_by = tf.reduce_max(over_by)
        overflowed = tf.reduce_any(max_over_by > 0)

        tf.assert_equal(
            overflowed,
            False,
            message=f"{message} Overflowed by {over_by} (positive number indicates overflow amount). Values should be less than {[tf.cast(t_half / 2 / s, grads[0].dtype) for s in scaling_factors]}",
        )

    def shell_train_step(self, features, labels, read_key_from_cache, apply_gradients):
        """
        The main training loop for the PostScale protocol.

        Args:
            features (tf.Tensor): Training features.
            labels (tf.Tensor): Training labels.
            read_key_from_cache: Boolean whether to read keys from cache.
            apply_gradients: Boolean whether to apply gradients.

        Returns:
            result: Dictionary of metric results.
            batch_size_should_be: Number of slots in encryption (ciphertext ring
                degree), or None if encryption is disabled.
        """
        with tf.device(self.labels_party_dev):
            labels = tf.cast(labels, tf.keras.backend.floatx())
            if self.disable_he_backprop_INSECURE:
                enc_labels = labels
            else:
                backprop_context = self.backprop_context_fn(read_key_from_cache)
                backprop_secret_key = hadal.create_key64(
                    backprop_context, read_key_from_cache, self.cache_path
                )
                # Encrypt the batch of secret labels.
                enc_labels = hadal.to_encrypted(
                    labels, backprop_secret_key, backprop_context
                )

        # Copy weights to the shadow models.
        for dev, dev_model in zip(self.jacobian_devices, self.jacobian_shadow_models):
            for dev_var, var in zip(dev_model.weights, self.weights):
                with tf.device(dev):
                    dev_var.assign(var)

        # Call the derived class to compute the gradients.
        grads, per_example_norms, predictions = self.compute_grads(features, enc_labels)

        with tf.device(self.features_party_dev):
            # Compute the maximum L2 norm of all per-example, per-class
            # gradients for noise sampling purposes.
            max_two_norm = tf.minimum(
                tf.cast(self.clip_threshold, tf.keras.backend.floatx()),
                tf.reduce_max(per_example_norms),
            )

            # Clip the gradients to the maximum L2 norm (only if adding noise)
            if self.noise_multiplier > 0.0:
                if self.simple_noise_INSECURE:
                    # If simple noise is enabled, clip the per-example gradients to
                    # the maximum L2 norm to mimic the DP-SGD algorithm.
                    grads = tf.vectorized_map(self.clip_gradient, grads)

                    # Override the max_two_norm, which is used to apply DP noise to
                    # the gradients, as the clipping threshold. This is how DP-SGD
                    # works, it samples DP noise assuming the gradients are always
                    # clipped to the threshold.
                    max_two_norm = self.clip_threshold
                else:
                    # Clip the gradients by the maximum per-class norm for each example.
                    clipper = tf.minimum(self.clip_threshold / per_example_norms, 1.0)

                    if self.disable_he_backprop_INSECURE:

                        def _plaintext_clip(grads_and_clipper):
                            c = grads_and_clipper[0]
                            gs = grads_and_clipper[1:]
                            return [g * c for g in gs]

                        grads.insert(0, clipper)
                        grads = tf.vectorized_map(_plaintext_clip, grads)
                    else:
                        # Since the clipping is done with multiplication, the
                        # scaling factor of the gradients will increase after
                        # clipping. The clipping scaling factor is limited
                        # below to reduce the growth. The extra precision is not
                        # needed.
                        clipping_scaling_factors = [g._scaling_factor for g in grads]
                        clipping_scaling_factors = [
                            min(c, backprop_context.scaling_factor)
                            for c in clipping_scaling_factors
                        ]
                        clipping_scaling_factors = [
                            max(c, 32.0) for c in clipping_scaling_factors
                        ]

                        # Compute the smallest rounding for the clipper. This
                        # ensures that the gradient norm is infact below the
                        # clipping threshold after quantization. Without this,
                        # the norm might be higher, the DP sensitivity might be
                        # underestimated, and privacy slightly weaker than
                        # stated.
                        rounded_clippers = [
                            hadal.smallest_case_rounding(clipper, csf)
                            for csf in clipping_scaling_factors
                        ]
                        rounded_clippers = [
                            hadal.to_shell_plaintext(
                                rc,
                                backprop_context,
                                override_scaling_factor=csf,
                                randomized_rounding_override=False,
                            )
                            for rc, csf in zip(
                                rounded_clippers, clipping_scaling_factors
                            )
                        ]

                        # Perform the clipping.
                        grads = [g * rc for g, rc in zip(grads, rounded_clippers)]

            # Store the scaling factors of the gradients.
            if self.disable_he_backprop_INSECURE:
                backprop_scaling_factors = [1] * len(grads)
            else:
                backprop_scaling_factors = [g._scaling_factor for g in grads]

            # Check if the backproped gradients overflowed.
            if not self.disable_he_backprop_INSECURE and self.check_overflow_INSECURE:
                # Note, checking the gradients requires decryption on the
                # features party which breaks security of the protocol.
                dec_grads = [hadal.to_tensorflow(g, backprop_secret_key) for g in grads]
                self.warn_on_overflow(
                    dec_grads,
                    backprop_scaling_factors,
                    tf.identity(backprop_context.plaintext_modulus),
                    "WARNING: Backprop gradient may have overflowed.",
                )

                clipped_grad_norm = tf.reduce_max(
                    tf.vectorized_map(self.gradient_norms, dec_grads)
                )
                self.warn_on_overflow(
                    [clipped_grad_norm],
                    [1.0],
                    (self.clip_threshold + 1e-3)
                    * 4,  # warn_on_overflow divides this by 4.
                    "WARNING: Clipped gradient norms exceed clipping threshold. DP sensitivity may be underestimated.",
                )

            # Spoof encrypted gradients as int64 with scaling factor 1 for
            # subsequent masking and noising, so the intermediate decryption
            # is not affected by the scaling factors.
            if not self.disable_he_backprop_INSECURE:
                grads = self.spoof_int_gradients(grads)

            # Mask the encrypted gradients.
            if not self.disable_masking_INSECURE:
                grads, masks = self.mask_gradients(backprop_context, grads)

            if not self.simple_noise_INSECURE:
                # Set up the features party side of the distributed noise
                # sampling sub-protocol.
                noise_context = self.noise_context_fn(read_key_from_cache)
                noise_secret_key = hadal.create_key64(
                    noise_context, read_key_from_cache, self.cache_path
                )

                # The noise context must have the same number of slots
                # (encryption ring degree) as used in backpropagation.
                if not self.disable_he_backprop_INSECURE:
                    tf.assert_equal(
                        tf.identity(backprop_context.num_slots),
                        noise_context.num_slots,
                        message="Backprop and noise contexts must have the same number of slots.",
                    )

                # The noise scaling factor must always be 1. Encryptions already
                # have the scaling factor applied when the noise is applied,
                # and an additional noise scaling factor is not needed.
                tf.assert_equal(
                    noise_context.scaling_factor,
                    1.0,
                    message="Noise scaling factor must be 1.",
                )

                # Compute the noise factors for the distributed noise sampling
                # sub-protocol.
                noise_factors = self.compute_noise_factors(
                    noise_context,
                    noise_secret_key,
                    max_two_norm,
                    backprop_scaling_factors,
                )

        with tf.device(self.labels_party_dev):
            if not self.disable_he_backprop_INSECURE:
                # Decrypt the weight gradients with the backprop key.
                grads = [hadal.to_tensorflow(g, backprop_secret_key) for g in grads]

            # Sum the masked gradients over the batch.
            if self.disable_masking_INSECURE:
                # No mask has been added so a only a normal sum is required.
                grads = [tf.reduce_sum(g, axis=0) for g in grads]
            else:
                grads = [
                    hadal.reduce_sum_with_mod(g, 0, backprop_context, 1) for g in grads
                ]

            if self.simple_noise_INSECURE:
                # If the noise protocol is disabled but the noise multiplier is
                # still greater than 0, add plaintext noise to the gradients.
                if self.noise_multiplier > 0.0:
                    # Note, generating the noise in plaintext is not secure, so
                    # we ignore quantization error wrt privacy when sampling
                    # from a floating point distribution.
                    noise = [
                        tf.random.normal(
                            shape=tf.shape(g),
                            mean=0.0,
                            stddev=self.noise_multiplier * max_two_norm * sf,
                            dtype=tf.keras.backend.floatx(),
                        )
                        for g, sf in zip(grads, backprop_scaling_factors)
                    ]
                    # When encryption and masking are enabled, the grads are
                    # integer types
                    noise = [tf.cast(n, dtype=g.dtype) for n, g in zip(noise, grads)]
                    grads = [g + n for g, n in zip(grads, noise)]
            else:
                # Efficiently pack the masked gradients to prepare for adding
                # the encrypted noise. This is special because the masked
                # gradients are no longer batched, so the packing must be done
                # manually.
                flat_grads, flat_metadata = self.flatten_and_pad_grad_list(
                    grads, tf.identity(noise_context.num_slots)
                )

                # Add the encrypted noise to the masked gradients.
                grads = self.noise_gradients(noise_context, flat_grads, noise_factors)

        with tf.device(self.features_party_dev):
            if not self.simple_noise_INSECURE:
                # The gradients must be first be decrypted using the noise
                # secret key.
                flat_grads = [hadal.to_tensorflow(g, noise_secret_key) for g in grads]

                # Unpack the noised grads after decryption.
                grads = self.unflatten_grad_list(flat_grads, flat_metadata)

            # Unmask the gradients.
            if not self.disable_masking_INSECURE:
                grads = self.unmask_gradients(backprop_context, grads, masks)

            # Recover the original scaling factor of the gradients if they were
            # originally encrypted.
            if not self.disable_he_backprop_INSECURE:
                grads = self.unspoof_int_gradients(grads, backprop_scaling_factors)

            if not self.simple_noise_INSECURE:
                if self.check_overflow_INSECURE:
                    self.warn_on_overflow(
                        grads,
                        [1] * len(grads),
                        noise_context.plaintext_modulus,
                        "WARNING: Noised gradient may have overflowed.",
                    )

            # When encryption is disabled but the noise protocol is enabled,
            # the gradients are ints. Convert them to floats.
            grads = [tf.cast(g, dtype=tf.keras.backend.floatx()) for g in grads]

            # Apply the gradients to the model.
            if not apply_gradients:
                # make grads very small but still apply them so the grappler
                # optimizer does not remove them.
                grads = [g / 1e9 for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            for metric in self.metrics:
                if metric.name == "loss":
                    if self.disable_he_backprop_INSECURE:
                        loss = self.compute_loss(y=labels, y_pred=predictions)
                        metric.update_state(loss)
                    else:
                        # Loss is unknown when encrypted.
                        metric.update_state(0.0)
                else:
                    if self.disable_he_backprop_INSECURE:
                        metric.update_state(labels, predictions)
                    else:
                        # Other metrics are uknown when encrypted.
                        zeros = tf.broadcast_to(0, tf.shape(predictions))
                        metric.update_state(zeros, zeros)

            metric_results = {m.name: m.result() for m in self.metrics}

            # TensorFlow 2.18.0 added a "CompiledMetrics" metric which holds metrics
            # passed to compile in it's own dictionary. Keras wants all metrics to
            # be returned as a flat dictionary. Here we flatten the dictionary.
            result = {}
            for key, value in metric_results.items():
                if isinstance(value, dict):
                    result.update(value)  # add subdict directly into the dict
                else:
                    result[key] = value  # non-subdict elements are just copied

            if not self.disable_he_backprop_INSECURE:
                ret_batch_size = tf.identity(backprop_context.num_slots)
            elif not self.simple_noise_INSECURE:
                ret_batch_size = tf.identity(noise_context.num_slots)
            else:
                ret_batch_size = None

            return result, ret_batch_size
