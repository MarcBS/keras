# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ..engine import Layer, InputSpec
from .. import initializers
from .. import regularizers
from .. import constraints
from .. import backend as K
from ..legacy import interfaces


def L1_norm(x):
    '''Computes the L1 norm of the input
    '''
    return K.l1_normalize(x, axis=1)

def L2_norm(x, axis=1):
    '''Computes the L2 norm of the input
    '''
    return K.l2_normalize(x, axis=axis)

def signed_sqrt(x):
    '''Signed square root of the input
    '''
    return K.switch(x >= 0, K.sqrt(x), -K.sqrt(-x))


class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BatchNormalization(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        momentum: Momentum for the moving average.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    @interfaces.legacy_batchnorm_support
    def __init__(self,
                 axis=-1,
                 mode=0,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):

        if self.mode == 1:
            self.input_spec = [InputSpec(shape=input_shape)]
            shape = (input_shape[self.axis],)
        else:
            dim = input_shape[self.axis]
            if dim is None:
                raise ValueError('Axis ' + str(self.axis) + ' of '
                                 'input tensor should have a defined dimension '
                                 'but the layer received an input with shape ' +
                                 str(input_shape) + '.')
            self.input_spec = InputSpec(ndim=len(input_shape),
                                        axes={self.axis: dim})
            shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        if self.mode == 0 or self.mode == 2:
            input_shape = K.int_shape(inputs)
            # Prepare broadcasting shape.
            ndim = len(input_shape)
            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            # Determines whether broadcasting is needed.
            needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

            def normalize_inference():
                if needs_broadcasting:
                    # In this case we must explicitly broadcast all parameters.
                    broadcast_moving_mean = K.reshape(self.moving_mean,
                                                      broadcast_shape)
                    broadcast_moving_variance = K.reshape(self.moving_variance,
                                                          broadcast_shape)
                    if self.center:
                        broadcast_beta = K.reshape(self.beta, broadcast_shape)
                    else:
                        broadcast_beta = None
                    if self.scale:
                        broadcast_gamma = K.reshape(self.gamma,
                                                    broadcast_shape)
                    else:
                        broadcast_gamma = None
                    return K.batch_normalization(
                        inputs,
                        broadcast_moving_mean,
                        broadcast_moving_variance,
                        broadcast_beta,
                        broadcast_gamma,
                        epsilon=self.epsilon)
                else:
                    return K.batch_normalization(
                        inputs,
                        self.moving_mean,
                        self.moving_variance,
                        self.beta,
                        self.gamma,
                        epsilon=self.epsilon)

            # If the learning phase is *static* and set to inference:
            if training in {0, False}:
                return normalize_inference()

            # If the learning is either dynamic, or set to training:
            normed_training, mean, variance = K.normalize_batch_in_training(
                inputs, self.gamma, self.beta, reduction_axes,
                epsilon=self.epsilon)

            self.add_update([K.moving_average_update(self.moving_mean,
                                                     mean,
                                                     self.momentum),
                             K.moving_average_update(self.moving_variance,
                                                     variance,
                                                     self.momentum)],
                            inputs)

            # Pick the normalized form corresponding to the training phase.
            return K.in_train_phase(normed_training,
                                    normalize_inference,
                                    training=training)
        elif self.mode == 1:
            # sample-wise normalization
            m = K.mean(inputs, axis=-1, keepdims=True)
            std = K.sqrt(K.var(inputs, axis=-1, keepdims=True) + self.epsilon)
            x_normed = (inputs - m) / (std + self.epsilon)
            x_normed = self.gamma * x_normed + self.beta
            return x_normed


    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'mode': self.mode,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer)}
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
