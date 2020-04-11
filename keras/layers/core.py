# -*- coding: utf-8 -*-
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import copy
import types as python_types
import warnings

from .. import backend as K
from .. import activations
from .. import initializers
from .. import regularizers
from .. import constraints
from ..engine.base_layer import InputSpec
from ..engine.base_layer import Layer
from ..utils.generic_utils import func_dump
from ..utils.generic_utils import func_load
from ..utils.generic_utils import deserialize_keras_object
from ..utils.generic_utils import has_arg
from ..legacy import interfaces


class Masking(Layer):
    """Masks a sequence by using a mask value to skip timesteps.

    If all features for a given sample timestep are equal to `mask_value`,
    then the sample timestep will be masked (skipped) in all downstream layers
    (as long as they support masking).

    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.

    # Example

    Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to an LSTM layer.
    You want to mask sample #0 at timestep #3, and sample #2 at timestep #5,
    because you lack features for these sample timesteps. You can do:

        - set `x[0, 3, :] = 0.` and `x[2, 5, :] = 0.`
        - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        model.add(LSTM(32))
    ```

    # Arguments
        mask_value: Either None or mask value to skip
    """

    def __init__(self, mask_value=0., **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_mask(self, inputs, mask=None):
        output_mask = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
        return output_mask

    def call(self, inputs):
        boolean_mask = K.any(K.not_equal(inputs, self.mask_value),
                             axis=-1, keepdims=True)
        return inputs * K.cast(boolean_mask, K.dtype(inputs))

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Dropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class GuidedDropout(Layer):

    """Applies a guided Dropout to the input, where the output activations are set
    to 0 given by the weights of the layer.
    # TODO: Test this layer

    # Arguments:
        weights_shape: (num_dropout_matrices, num_features)
        weights: Initial weights
    """

    def __init__(self, weights_shape, weights=None, **kwargs):
        self.weights_shape = weights_shape
        self.initial_weights = [weights]
        self.init = initializers.get('uniform', dim_ordering='th')
        super(GuidedDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init(self.weights_shape,
                           name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]

        # initialize weights
        if (self.initial_weights[0] is not None):
            self.set_weights(self.initial_weights)

        self.trainable = False

    def call(self, inputs, mask=None):
        modulated_input = inputs[0]
        modulator_input = inputs[1]

        modulated_output = modulated_input * self.W[K.argmax(modulator_input, axis=1), :]

        return modulated_output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {'weights_shape': self.weights_shape}
        base_config = super(GuidedDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatialDropout1D(Dropout):
    """Spatial 1D version of Dropout.

    This version performs the same function as Dropout, however it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout1D will help promote independence
    between feature maps and should be used instead.

    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - [Efficient Object Localization Using Convolutional Networks](
           https://arxiv.org/abs/1411.4280)
    """

    @interfaces.legacy_spatialdropout1d_support
    def __init__(self, rate, **kwargs):
        super(SpatialDropout1D, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], 1, input_shape[2])
        return noise_shape


class SpatialDropout2D(Dropout):
    """Spatial 2D version of Dropout.

    This version performs the same function as Dropout, however it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout2D will help promote independence
    between feature maps and should be used instead.

    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        data_format: `'channels_first'` or `'channels_last'`.
            In `'channels_first'` mode, the channels dimension
            (the depth) is at index 1,
            in `'channels_last'` mode is it at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `'channels_last'`.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if `data_format='channels_first'`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if `data_format='channels_last'`.

    # Output shape
        Same as input

    # References
        - [Efficient Object Localization Using Convolutional Networks](
           https://arxiv.org/abs/1411.4280)
    """

    @interfaces.legacy_spatialdropoutNd_support
    def __init__(self, rate, data_format=None, **kwargs):
        super(SpatialDropout2D, self).__init__(rate, **kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, input_shape[3])
        return noise_shape


class SpatialDropout3D(Dropout):
    """Spatial 3D version of Dropout.

    This version performs the same function as Dropout, however it drops
    entire 3D feature maps instead of individual elements. If adjacent voxels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout3D will help promote independence
    between feature maps and should be used instead.

    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        data_format: `'channels_first'` or `'channels_last'`.
            In `'channels_first'` mode, the channels dimension (the depth)
            is at index 1, in `'channels_last'` mode is it at index 4.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `'channels_last'`.

    # Input shape
        5D tensor with shape:
        `(samples, channels, dim1, dim2, dim3)` if `data_format='channels_first'`
        or 5D tensor with shape:
        `(samples, dim1, dim2, dim3, channels)` if `data_format='channels_last'`.

    # Output shape
        Same as input

    # References
        - [Efficient Object Localization Using Convolutional Networks](
           https://arxiv.org/abs/1411.4280)
    """

    @interfaces.legacy_spatialdropoutNd_support
    def __init__(self, rate, data_format=None, **kwargs):
        super(SpatialDropout3D, self).__init__(rate, **kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1, 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, 1, input_shape[4])
        return noise_shape


class Activation(Layer):
    """Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Reshape(Layer):
    """Reshapes an output to a certain shape.

    # Arguments
        target_shape: target shape. Tuple of integers.
            Does not include the batch axis.

    # Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis)
        when using this layer as the first layer in a model.

    # Output shape
        `(batch_size,) + target_shape`

    # Example

    ```python
        # as first layer in a Sequential model
        model = Sequential()
        model.add(Reshape((3, 4), input_shape=(12,)))
        # now: model.output_shape == (None, 3, 4)
        # note: `None` is the batch dimension

        # as intermediate layer in a Sequential model
        model.add(Reshape((6, 2)))
        # now: model.output_shape == (None, 6, 2)

        # also supports shape inference using `-1` as dimension
        model.add(Reshape((-1, 2, 2)))
        # now: model.output_shape == (None, 3, 2, 2)
    ```
    """

    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Finds and replaces a missing dimension in an output shape.

        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

        # Arguments
            input_shape: original shape of array being reshaped
            output_shape: target shape of the array, with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.

        # Returns
            The new output shape with a `-1` replaced with its computed value.

        # Raises
            ValueError: if `input_shape` and `output_shape` do not match.
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        if None in input_shape[1:]:
            # input shape (partially) unknown? replace -1's with None's
            return ((input_shape[0],) +
                    tuple(s if s != -1 else None for s in self.target_shape))
        else:
            # input shape known? then we can compute the output shape
            return (input_shape[0],) + self._fix_unknown_dimension(
                input_shape[1:], self.target_shape)

    def call(self, inputs):
        return K.reshape(inputs, (K.shape(inputs)[0],) + self.target_shape)

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern.

    Useful for e.g. connecting RNNs and convnets together.

    # Example

    ```python
        model = Sequential()
        model.add(Permute((2, 1), input_shape=(10, 64)))
        # now: model.output_shape == (None, 64, 10)
        # note: `None` is the batch dimension
    ```

    # Arguments
        dims: Tuple of integers. Permutation pattern, does not include the
            samples dimension. Indexing starts at 1.
            For instance, `(2, 1)` permutes the first and second dimension
            of the input.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.
    """

    def __init__(self, dims, **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.dims = tuple(dims)
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i + 1] = target_dim
        return tuple(output_shape)

    def call(self, inputs):
        return K.permute_dimensions(inputs, (0,) + self.dims)

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(Permute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PermuteGeneral(Layer):
    """Permutes the dimensions of the input according to a given pattern.
    This is just like the layer Permute, but DOES INCLUDE the batch dimension.

    # Arguments
        dims: Tuple of integers. Permutation pattern, INCLUDING the
            samples dimension. Indexing starts at 0.
            For instance, `(1, 0, 2)` permutes the batch and first dimension of the input.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.
    """

    def __init__(self, dims, **kwargs):
        super(PermuteGeneral, self).__init__(**kwargs)
        self.dims = tuple(dims)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            output_shape[i] = input_shape[dim]
        return tuple(output_shape)

    def call(self, inputs, mask=None):
        return K.permute_dimensions(inputs, self.dims)

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(PermuteGeneral, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    # Arguments
        data_format: A string,
            one of `'channels_last'` (default) or `'channels_first'`.
            The ordering of the dimensions in the inputs.
            The purpose of this argument is to preserve weight
            ordering when switching a model from one data format
            to another.
            `'channels_last'` corresponds to inputs with shape
            `(batch, ..., channels)` while `'channels_first'` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `'channels_last'`.

    # Example

    ```python
        model = Sequential()
        model.add(Conv2D(64, (3, 3),
                         input_shape=(3, 32, 32), padding='same',))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.data_format = K.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '). '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, inputs):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)

        return K.batch_flatten(inputs)

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RepeatVector(Layer):
    """Repeats the input n times.

    # Example

    ```python
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        # now: model.output_shape == (None, 32)
        # note: `None` is the batch dimension

        model.add(RepeatVector(3))
        # now: model.output_shape == (None, 3, 32)
    ```

    # Arguments
        n: integer, repetition factor.

    # Input shape
        2D tensor of shape `(num_samples, features)`.

    # Output shape
        3D tensor of shape `(num_samples, n, features)`.
    """

    def __init__(self, n, **kwargs):
        super(RepeatVector, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def call(self, inputs):
        return K.repeat(inputs, self.n)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RepeatMatrix(Layer):
    """Repeats the input n times.
       Applies the same procedure as RepeatVector() but for inputs of any dimenions.
       The new dimension will be introduced in the position defined by the user.

    # Arguments
        n: integer, repetition factor.
        dim: integer, dimension along which the input will be repeated (default = 1)

    # Input shape
        R-dimensional tensor of shape `(nb_samples, dim1, dim2, ..., dimR-1)`.

    # Output shape
        R+1-dimensional tensor of shape `(nb_samples, n, dim2, dim3, ..., dimR)` if dim==1.
    """

    def __init__(self, n, dim=1, **kwargs):
        self.supports_masking = True
        self.n = n
        self.dim = dim
        super(RepeatMatrix, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[:self.dim]) + [self.n] + list(input_shape[self.dim:])
        return tuple(output_shape)

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def call(self, inputs, mask=None):
        return K.repeatRdim(inputs, self.n, axis=self.dim)

    def get_config(self):
        config = {'n': self.n,
                  'dim': self.dim}
        base_config = super(RepeatMatrix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Lambda(Layer):
    """Wraps arbitrary expression as a `Layer` object.

    # Examples

    ```python
        # add a x -> x^2 layer
        model.add(Lambda(lambda x: x ** 2))
    ```
    ```python
        # add a layer that returns the concatenation
        # of the positive part of the input and
        # the opposite of the negative part

        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)

        def antirectifier_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        model.add(Lambda(antirectifier,
                         output_shape=antirectifier_output_shape))
    ```
    ```python
        # add a layer that returns the hadamard product
        # and sum of it from two input tensors

        def hadamard_product_sum(tensors):
            out1 = tensors[0] * tensors[1]
            out2 = K.sum(out1, axis=-1)
            return [out1, out2]

        def hadamard_product_sum_output_shape(input_shapes):
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            assert shape1 == shape2  # else hadamard product isn't possible
            return [tuple(shape1), tuple(shape2[:-1])]

        x1 = Dense(32)(input_1)
        x2 = Dense(32)(input_2)
        layer = Lambda(hadamard_product_sum, hadamard_product_sum_output_shape)
        x_hadamard, x_sum = layer([x1, x2])
    ```

    # Arguments
        function: The function to be evaluated.
            Takes input tensor or list of tensors as first argument.
        output_shape: Expected output shape from function.
            Only relevant when using Theano.
            Can be a tuple or function.
            If a tuple, it only specifies the first dimension onward;
                 sample dimension is assumed either the same as the input:
                 `output_shape = (input_shape[0], ) + output_shape`
                 or, the input is `None` and
                 the sample dimension is also `None`:
                 `output_shape = (None, ) + output_shape`
            If a function, it specifies the entire shape as a function of the
            input shape: `output_shape = f(input_shape)`
        mask: Either None (indicating no masking) or a Tensor indicating the
          input mask for Embedding.
        arguments: optional dictionary of keyword arguments to be passed
            to the function.
        supports_masking: whether it supports masking or not

    # Input shape
        Arbitrary. Use the keyword argument input_shape
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Specified by `output_shape` argument
        (or auto-inferred when using TensorFlow or CNTK).
    """

    @interfaces.legacy_lambda_support
    def __init__(self, function, output_shape=None,
                 mask=None, arguments=None,
                 supports_masking=True, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.function = function
        self._input_dtypes = None
        self.arguments = arguments if arguments else {}
        self.supports_masking = supports_masking
        if mask is not None:
            self.supports_masking = True
        self.mask = mask

        if output_shape is None:
            self._output_shape = None
        elif isinstance(output_shape, (tuple, list)):
            self._output_shape = tuple(output_shape)
        else:
            if not callable(output_shape):
                raise TypeError('In Lambda, `output_shape` '
                                'must be a list, a tuple, or a function.')
            self._output_shape = output_shape

        super(Lambda, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self._output_shape is None:
            # With TensorFlow or CNTK, we can infer the output shape directly:
            if K.backend() in ('tensorflow', 'cntk'):
                if isinstance(input_shape, list):
                    xs = [K.placeholder(shape=shape, dtype=dtype)
                          for shape, dtype in zip(input_shape, self._input_dtypes)]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape, dtype=self._input_dtypes)
                    x = self.call(x)
                if isinstance(x, list):
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # Otherwise, we default to the input shape.
            warnings.warn('`output_shape` argument not specified for layer {} '
                          'and cannot be automatically inferred '
                          'with the Theano backend. '
                          'Defaulting to output shape `{}` '
                          '(same as input shape). '
                          'If the expected output shape is different, '
                          'specify it via the `output_shape` argument.'
                          .format(self.name, input_shape))
            return input_shape
        elif isinstance(self._output_shape, (tuple, list)):
            if isinstance(input_shape, list):
                num_samples = input_shape[0][0]
            else:
                num_samples = input_shape[0] if input_shape else None
            return (num_samples,) + tuple(self._output_shape)
        else:
            shape = self._output_shape(input_shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError('`output_shape` function must return a tuple or '
                                 'a list of tuples.')
            if isinstance(shape, list):
                if isinstance(shape[0], int) or shape[0] is None:
                    shape = tuple(shape)
            return shape

    def call(self, inputs, mask=None):
        arguments = self.arguments
        if has_arg(self.function, 'mask'):
            arguments['mask'] = mask
        if isinstance(inputs, list):
            self._input_dtypes = [K.dtype(x) for x in inputs]
        else:
            self._input_dtypes = K.dtype(inputs)
        return self.function(inputs, **arguments)

    def compute_mask(self, inputs, mask=None):
        if callable(self.mask):
            return self.mask(inputs, mask)
        return self.mask

    def get_config(self):
        if isinstance(self.function, python_types.LambdaType):
            function = func_dump(self.function)
            function_type = 'lambda'
        else:
            function = self.function.__name__
            function_type = 'function'

        if isinstance(self._output_shape, python_types.LambdaType):
            output_shape = func_dump(self._output_shape)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        config = {'function': function,
                  'function_type': function_type,
                  'output_shape': output_shape,
                  'output_shape_type': output_shape_type,
                  'arguments': self.arguments}
        base_config = super(Lambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        globs = globals()
        if custom_objects:
            globs = dict(list(globs.items()) + list(custom_objects.items()))
        function_type = config.pop('function_type')
        if function_type == 'function':
            # Simple lookup in custom objects
            function = deserialize_keras_object(
                config['function'],
                custom_objects=custom_objects,
                printable_module_name='function in Lambda layer')
        elif function_type == 'lambda':
            # Unsafe deserialization from bytecode
            function = func_load(config['function'], globs=globs)
        else:
            raise TypeError('Unknown function type:', function_type)

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            # Simple lookup in custom objects
            output_shape = deserialize_keras_object(
                config['output_shape'],
                custom_objects=custom_objects,
                printable_module_name='output_shape function in Lambda layer')
        elif output_shape_type == 'lambda':
            # Unsafe deserialization from bytecode
            output_shape = func_load(config['output_shape'], globs=globs)
        else:
            output_shape = config['output_shape']

        # If arguments were numpy array, they have been saved as
        # list. We need to recover the ndarray
        if 'arguments' in config:
            for key in config['arguments']:
                if isinstance(config['arguments'][key], dict):
                    arg_dict = config['arguments'][key]
                    if 'type' in arg_dict and arg_dict['type'] == 'ndarray':
                        # Overwrite the argument with its numpy translation
                        config['arguments'][key] = np.array(arg_dict['value'])

        config['function'] = function
        config['output_shape'] = output_shape
        return cls(**config)


class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ActivityRegularization(Layer):
    """Layer that applies an update to the cost function based input activity.

    # Arguments
        l1: L1 regularization factor (positive float).
        l2: L2 regularization factor (positive float).

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, l1=0., l2=0., **kwargs):
        super(ActivityRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2
        self.activity_regularizer = regularizers.L1L2(l1=l1, l2=l2)

    def get_config(self):
        config = {'l1': self.l1,
                  'l2': self.l2}
        base_config = super(ActivityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionLayer(Layer):
    """
    This layer is returns a sequence as the input, but as a range.
    It recieves an  input with shape: (B, N) and returns a tensor with the same
    shape but with the content:
    [arange(N), arange(N), ..., arange(N)]

    # Arguments
        maxlen: Integer, maximum length allowed (N)
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, maxlen=500, **kwargs):
        self.supports_masking = True
        self.positions = K.arange(0, stop=maxlen)
        super(PositionLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return self.positions[:K.shape(inputs)[1]]

    def compute_mask(self, input_shape, input_mask=None):
        return input_mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(PositionLayer, self).get_config()
        return dict(list(base_config.items()))


class RemoveThirdDimension(Layer):
    """
    Removes the third dimension of a 3D tensor.

    # Input shape
        3D. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input but without the last dimension.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(RemoveThirdDimension, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return inputs[:, :, 0]

    def compute_mask(self, input_shape, input_mask=None):
        return input_mask

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        base_config = super(RemoveThirdDimension, self).get_config()
        return dict(list(base_config.items()))


class MaskedMean(Layer):
    """
    This layer is called after an Embedding layer.
    It averages all of the masked-out embeddings.
    The mask is discarded.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedMean, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return K.mean(K.cast(mask[:, :, None], K.dtype(inputs)) * inputs, axis=1)

    def compute_mask(self, input_shape, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        base_config = super(MaskedMean, self).get_config()
        return dict(list(base_config.items()))


class ApplyMask(Layer):
    """
    This layer is called after an Embedding layer.
    It averages all of the masked-out embeddings.
    The mask is discarded.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(ApplyMask, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        mask = inputs[1]
        inputs = inputs[0]
        return K.cast(mask[:, :, None], K.dtype(inputs)) * inputs

    def compute_mask(self, inputs, mask=None):
        return inputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super(ApplyMask, self).get_config()
        return dict(list(base_config.items()))


class MaskLayer(Layer):
    """Applies to the input its mask. The mask is kept.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return K.cast(mask[:, :, None], K.dtype(inputs)) * inputs

    def compute_mask(self, input_shape, input_mask=None):
        return input_mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(MaskLayer, self).get_config()
        return dict(list(base_config.items()))


class MaskAndRemoveMask(Layer):
    """Applies to the input its mask. The mask is removed.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskAndRemoveMask, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return K.cast(mask[:, :, None], K.dtype(inputs)) * inputs

    def compute_mask(self, input_shape, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(MaskAndRemoveMask, self).get_config()
        return dict(list(base_config.items()))


class RemoveMask(Layer):
    """Removes the mask of the Layer.
    """

    def __init__(self, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        base_config = super(RemoveMask, self).get_config()
        return dict(list(base_config.items()))


class GetMask(Layer):
    """Gets the mask of a layer.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GetMask, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return mask

    def compute_mask(self, input_shape, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(GetMask, self).get_config()
        return dict(list(base_config.items()))


class FlatMask(Layer):
    """
    Flattens a n-dimensional mask to an (n-1)-dimensional one.

    # Arguments
        axis: Axis to flatten the mask.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, axis=2, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(FlatMask, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return inputs

    def compute_mask(self, input_shape, input_mask=None):
        return K.any(input_mask, self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(FlatMask, self).get_config()
        return dict(list(base_config.items()))


class WeightedSum(Layer):
    """ Applies a weighted sum over a set of vectors input[0] and their respective weights input[1].
        First, the weights are tiled for matching the length of the input vectors on dim=1.
        Second, an element-wise multiplication is applied over the inputs.
        Third, the output tensor is summed over the defined set of dimensions if
        the input parameter sum_dims is provided.

    # Arguments
        sum_dims: dimensions on which the final summation will be applied after the respective multiplication

    # Input shape
        List with two tensors:
            input[0]: vectors
            input[1]: weights
        Both tensors must have a matching number of dimensions and lengths, except
        dim=1, which must be 1 for the set of weights.

    # Output shape
        Vector with the same number of dimensions and length as input[0] but having removed the dimensions
        specified in sum_dims (if any).
    """

    def __init__(self, sum_dims=[], **kwargs):
        assert isinstance(sum_dims, list)
        self.sum_dims = sorted(sum_dims)[::-1]
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2

    def call(self, inputs, mask=None):
        # get input values and weights
        values = inputs[0]
        weights = inputs[1]

        # tile weights before summing
        K.repeatRdim(weights, K.shape(values)[1], axis=1)

        # inputs = K.dot(values, weights)
        inputs = values * weights

        for d in self.sum_dims:
            inputs = K.sum(inputs, axis=d)
        return inputs

    def compute_output_shape(self, input_shape):
        out_dim = []
        num_dim = len(input_shape[0])
        for d in range(num_dim):
            if d not in self.sum_dims:
                out_dim.append(max(input_shape[0][d], input_shape[1][d]))
        return tuple(out_dim)

    def compute_mask(self, input, input_mask=None):
        if not any(input_mask):
            return None
        else:
            not_None_masks = [m for m in input_mask if m is not None]
            if len(not_None_masks) == 1:
                out_mask = input_mask[not_None_masks[0]]
            else:
                out_mask = input_mask[not_None_masks[0]] * input_mask[not_None_masks[1]]

            return out_mask

    def get_config(self):
        config = {'sum_dims': self.sum_dims}
        base_config = super(WeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedMerge(Layer):
    """ Applies a weighted merge over a set of tensors.
        This layer learns a set of lambda weights for applying a weighted sum
        for merging the input tensors.

    # Arguments
        mode: merge mode used. Possible values are 'sum' (default) or 'mul'.
        init: Initialization function.
        lambdas_regularizer: Regularizers for the weights.
        weights: Initial weights for each element to merge.

    # Input shape
        List of tensors of any dimensions but with the same shape.

    # Output shape
        Tensor with the same number of dimensions as the input tensors.
    """

    def __init__(self, mode='sum',
                 init='glorot_uniform',
                 lambdas_regularizer=None,
                 weights=None, **kwargs):
        # self.out_shape = out_shape
        self._valid_modes = ['sum', 'mul']

        if mode not in self._valid_modes:
            raise NotImplementedError(
                "Merge mode of type '" + mode + "' is not valid. Valid modes are: " + str(self._valid_modes))
        self.mode = mode

        self.init = initializers.get(init)
        self.lambdas_regularizer = regularizers.get(lambdas_regularizer)
        self.initial_weights = weights

        self.supports_masking = True
        super(WeightedMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        # Cnvert input_shape to list
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        s = input_shape[0]
        # Check all input_shape are compatible
        for i in range(1, len(input_shape)):
            for s1, s2 in zip(input_shape[i], s):
                assert s1 == s2 or s1 is None or s2 is None, 'The shapes of some input tensors do not match ' \
                                                             '(' + str(input_shape[i]) + ' vs ' + str(s) + ').'
        # Initialize weights
        self.lambdas = self.init((len(input_shape),), name='{}_lambdas'.format(self.name))
        # Set weights to trainable
        self.trainable_weights = [self.lambdas]
        # List of regularizers
        self.regularizers = []

        # Add lambdas regularizers (if necessary)
        if self.lambdas_regularizer:
            self.lambdas_regularizer.set_param(self.lambdas)
            self.regularizers.append(self.lambdas_regularizer)
        # Set initial weights (if necessary)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list):
            inputs = [inputs]

        # merge inputs after weighting by the learned lambda weights
        s = inputs[0] * self.lambdas[0]
        for i in range(1, len(inputs)):
            if self.mode == 'sum':
                s += inputs[i] * self.lambdas[i]
            elif self.mode == 'mul':
                s *= inputs[i] * self.lambdas[i]

        return s

    def compute_output_shape(self, input_shape):
        # return tuple(list(input_shape[0][:2]) + self.out_shape)

        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        return tuple(input_shape[0])

    def compute_mask(self, input, input_mask=None):
        if not isinstance(input_mask, list):
            input_mask = [input_mask]
        if not any(input_mask):
            return None
        else:
            return input_mask[0]

    def get_config(self):
        config = {'mode': self.mode,
                  'kernel_initializer': self.init.__name__,
                  'lambdas_regularizer': self.lambdas_regularizer.get_config() if self.lambdas_regularizer else None}
        base_config = super(WeightedMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SetSubtensor(Layer):
    """This layer performs a set_subtensor operation over two layers
    # Arguments
        indices: list of strings specifying the indexation over the two input layers

    # Input shape
        List with two tensors:
            input[0]: Tensor to overwrite
            input[1]: Tensor that overwrites
    # Output shape
        K.set_subtensor(input[0][indices[0], input[1][indices[1]])
    # Supports masking: The mask of the first input layer
    """

    def __init__(self, indices, **kwargs):
        self.supports_masking = True
        self.indices = indices
        super(SetSubtensor, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2

    def call(self, inputs, mask=None):
        return K.set_subtensor(eval('inputs[0]' + self.indices[0]), eval('inputs[1]' + self.indices[1]))

    def compute_mask(self, input_shape, input_mask=None):
        return input_mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {'indices': self.indices}
        base_config = super(SetSubtensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZeroesLayer(Layer):
    """Given any input, produces an output input_dim zeroes

    # Example

    ```python
        # as first

    # Arguments
        output_dim: int > 0.
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., units)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, units)`.
    """

    def __init__(self, output_dim, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(ZeroesLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs, mask=None):
        initial_state = K.zeros_like(inputs)  # (samples, input_dim)
        initial_state = K.sum(initial_state, axis=(1,))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])
        return initial_state

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'units': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ZeroesLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def set_lr_multipliers(self, W_learning_rate_multiplier, b_learning_rate_multiplier):
        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,
                                          self.b_learning_rate_multiplier]


class EqualDimensions(Layer):
    """Zero-padding layer for 2D input (e.g. picture).

    # Arguments
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, rows+1, cols+1)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows+1, cols+1, channels)` if dim_ordering='tf'.
    """

    def __init__(self,
                 dim_ordering='default',
                 **kwargs):
        super(EqualDimensions, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        self.dim_ordering = dim_ordering

    def compute_output_shape(self, input_shape):
        assert len(input_shape[0]) == len(input_shape[1])

        out_dims = [input_shape[1][0], input_shape[1][1], input_shape[0][2], input_shape[0][3]]
        return tuple(out_dims)

    def call(self, inputs, mask=None):
        return K.equal_dimensions(inputs[0], inputs[1])

    def get_config(self):
        config = {}
        base_config = super(EqualDimensions, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Concat(Layer):
    """Concatenates multiple inputs along the specified axis. Inputs should have the same
    shape except for the dimension specified in axis, which can have different sizes.

    # Arguments

        axis: int
            Axis which inputs are joined over

        cropping: None or [crop]
            Cropping for each input axis. Cropping is always disable for axis.

        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
    """

    def __init__(self, axis=1,
                 cropping=None, dim_ordering='default',
                 **kwargs):
        super(Concat, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')

        self.dim_ordering = dim_ordering
        self.axis = axis

        if cropping is not None:
            # If cropping is enabled, don't crop on the selected axis
            cropping = list(cropping)
            cropping[axis] = None
        self.cropping = cropping

    def compute_output_shape(self, input_shape):
        input_shapes = autocrop_array_shapes(input_shape, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
            return (len(shape1) == len(shape2) and
                    all(i == axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")

        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def call(self, inputs, mask=None):
        inputs = autocrop(inputs, self.cropping)
        return K.concatenate(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis, 'cropping': self.cropping}
        base_config = super(Concat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionwiseFeedForwardDense(Layer):
    """Fully connected feed-forward netwok, applied to each position.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(inputs) = inputs`).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation='relu',
                 dropout=0.,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PositionwiseFeedForwardDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel1 = self.add_weight(shape=(input_dim, self.units),
                                       initializer=self.kernel_initializer,
                                       name='kernel1',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias1 = self.add_weight(shape=(self.units,),
                                         initializer=self.bias_initializer,
                                         name='bias1',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        else:
            self.bias_1 = None

        self.kernel2 = self.add_weight(shape=(self.units, input_dim),
                                       initializer=self.kernel_initializer,
                                       name='kernel2',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias2 = self.add_weight(shape=(input_dim,),
                                         initializer=self.bias_initializer,
                                         name='bias2',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        else:
            self.bias_2 = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        if self.dropout > 0:
            self.dropout_layer = Dropout(self.dropout)
        self.built = True

    def call(self, inputs, mask=None):
        intermediate_output = K.dot(inputs, self.kernel1)
        if self.use_bias:
            intermediate_output = K.bias_add(intermediate_output, self.bias1)
        if self.activation is not None:
            intermediate_output = self.activation(intermediate_output)
        if self.dropout > 0:
            intermediate_output = self.dropout_layer(intermediate_output)
        output = K.dot(intermediate_output, self.kernel2)
        if self.use_bias:
            output = K.bias_add(output, self.bias2)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        return input_shape

    def compute_mask(self, input_shape, input_mask=None):
        return input_mask

    def get_config(self):
        config = {
            'units': self.units,
            'dropout': self.dropout,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(PositionwiseFeedForwardDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Slice(Layer):
    """Slices the input with the provided min-max indices.

    # Arguments
        min_idx: Integer indicating the start indices of the slice
        max_idx: Integer final indices of the slice

    # Input shape
        2D tensor

    # Output shape
        Same shape as input.
    """

    def __init__(self, min_idx, max_idx, **kwargs):
        self.supports_masking = True
        super(Slice, self).__init__(**kwargs)
        self.min_idx = min_idx
        self.max_idx = max_idx

    def call(self, inputs, mask=None):
        return inputs[:, self.min_idx:self.max_idx]

    def compute_mask(self, input_shape, input_mask=None):
        if input_mask is not None:
            return input_mask[:, self.min_idx:self.max_idx]
        else:
            return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_idx - self.min_idx)

    def get_config(self):
        config = {
            'min_idx': self.min_idx,
            'max_idx': self.max_idx
        }
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def autocrop(inputs, cropping):
    """Crops the given input arrays.

    Cropping takes a sequence of inputs and crops them per-axis in order to
    ensure that their sizes are consistent so that they can be combined
    in an element-wise fashion. If cropping is enabled for a specific axis,
    the minimum size in that axis of all inputs is computed, and all
    inputs are cropped to that size.

    The per-axis cropping modes are:

    `None`: this axis is not cropped, inputs are unchanged in this axis

    `'lower'`: inputs are cropped choosing the lower portion in this axis
    (`a[:crop_size, ...]`)

    `'upper'`: inputs are cropped choosing the upper portion in this axis
    (`a[-crop_size:, ...]`)

    `'center'`: inputs are cropped choosing the central portion in this axis
    (``a[offset:offset+crop_size, ...]`` where
    ``offset = (a.shape[0]-crop_size)//2)``

    # Arguments
    inputs: list of expressions
        The input arrays in the form of a list of Theano expressions
    cropping: list of cropping modes
        Cropping modes, one for each axis. If length of `cropping` is less
        than the number of axes in the inputs, it is padded with `None`.
        If `cropping` is None, `input` is returned as is.

    # Returns
        list of expressions. Each expression is the cropped version of the corresponding input
    """
    if cropping is None:
        # No cropping in any dimension
        return inputs
    else:
        # Get the number of dimensions
        ndim = K.ndim(inputs[0])

        # Check for consistent number of dimensions
        if not all(K.ndim(input) == ndim for input in inputs):
            raise ValueError("Not all inputs are of the same ",
                             "dimensionality. Got {0} inputs of "
                             "dimensionalities {1}.".format(
                                 len(inputs), [K.ndim(input) for input in inputs]))

        # Get the shape of each input
        shapes = [K.shape(input) for input in inputs]
        # Convert the shapes to a matrix expression
        shapes_tensor = K.variable(shapes)
        # Min along axis 0 to get the minimum size in each dimension
        min_shape = K.min(shapes_tensor, axis=0)

        # Nested list of slices; each list in `slices` corresponds to
        # an input and contains a slice for each dimension
        slices_by_input = [[] for i in range(len(inputs))]

        # If there are more dimensions than cropping entries, pad
        # the cropping
        cropping = list(cropping)
        if ndim > len(cropping):
            cropping = list(cropping) + [None] * (ndim - len(cropping))

        # For each dimension
        for dim, cr in enumerate(cropping):
            if cr is None:
                # Don't crop this dimension
                slice_all = slice(None)
                for slices in slices_by_input:
                    slices.append(slice_all)
            else:
                # We crop all inputs in the dimension `dim` so that they
                # are the minimum found in this dimension from all inputs
                sz = min_shape[dim]
                if cr == 'lower':
                    # Choose the first `sz` elements
                    slc_lower = slice(None, sz)
                    for slices in slices_by_input:
                        slices.append(slc_lower)
                elif cr == 'upper':
                    # Choose the last `sz` elements
                    slc_upper = slice(-sz, None)
                    for slices in slices_by_input:
                        slices.append(slc_upper)
                elif cr == 'center':
                    # Choose `sz` elements from the center
                    for sh, slices in zip(shapes, slices_by_input):
                        offset = (sh[dim] - sz) // 2
                        slices.append(slice(offset, offset + sz))
                else:
                    raise ValueError(
                        'Unknown crop mode \'{0}\''.format(cr))

        return [input[slices] for input, slices in
                zip(inputs, slices_by_input)]


def autocrop_array_shapes(input_shapes, cropping):
    """Computes the shapes of the given arrays after auto-cropping is applied.

    For more information on cropping, see the :func:`autocrop` function
    documentation.

    # Arguments
        input_shapes: the shapes of input arrays prior to cropping in
            the form of a list of tuples
        cropping: a list of cropping modes, one for each axis. If length of
            `cropping` is less than the number of axes in the inputs, it is
            padded with `None`. If `cropping` is None, `input_shapes` is returned
            as is. For more information on their values and operation, see the
            :func:`autocrop` documentation.

    # Returns
         shapes of the given arrays after auto-cropping is applied.
    """
    if cropping is None:
        return input_shapes
    else:
        # Check for consistent number of dimensions
        ndim = len(input_shapes[0])
        if not all(len(sh) == ndim for sh in input_shapes):
            raise ValueError("Not all inputs are of the same "
                             "dimensionality. Got {0} inputs of "
                             "dimensionalities {1}.".format(len(input_shapes), [len(sh) for sh in input_shapes]))

        result = []

        # If there are more dimensions than cropping entries, pad
        # the cropping
        cropping = list(cropping)
        if ndim > len(cropping):
            cropping = list(cropping) + [None] * (ndim - len(cropping))

        for sh, cr in zip(zip(*input_shapes), cropping):
            if cr is None:
                result.append(sh)
            elif cr in {'lower', 'center', 'upper'}:
                min_sh = None if any(inputs is None for inputs in sh) else min(sh)
                result.append([min_sh] * len(sh))
            else:
                raise ValueError('Unknown crop mode \'{0}\''.format(cr))
        return [tuple(sh) for sh in zip(*result)]
