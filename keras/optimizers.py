"""Built-in optimizer classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
import numpy as np
import warnings
from six.moves import zip

from . import backend as K
from .utils.generic_utils import serialize_keras_object
from .utils.generic_utils import deserialize_keras_object
from .legacy import interfaces

if K.backend() == 'tensorflow':
    import tensorflow as tf


def clip_norm(g, c, n):
    """Clip the gradient `g` if the L2 norm `n` exceeds `c`.

    # Arguments
        g: Tensor, the gradient tensor
        c: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        n: Tensor, actual norm of `g`.

    # Returns
        Tensor, the gradient clipped if required.
    """
    if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
        return g

    # tf require using a special op to multiply IndexedSliced by scalar
    if K.backend() == 'tensorflow':
        condition = n >= c
        then_expression = tf.scalar_mul(c / n, g)
        else_expression = g

        # saving the shape to avoid converting sparse tensor to dense
        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        g = tf.cond(condition,
                    lambda: then_expression,
                    lambda: else_expression)
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape
    else:
        g = K.switch(K.greater_equal(n, c), g * c / n, g)
    return g


class Optimizer(object):
    """Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    All Keras optimizers support the following keyword arguments:

        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        raise NotImplementedError

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if any(x is None for x in grads):
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). '
                             'Common ops without gradient: '
                             'K.argmax, K.round, K.eval.')
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.

        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        if len(params) != len(weights):
            raise ValueError('Length of the specified weight list (' +
                             str(len(weights)) +
                             ') does not match the number of weights ' +
                             'of the optimizer (' + str(len(params)) + ')')
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Optimizer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def set_lr(self, lr):
        K.set_value(self.lr, lr)

    def get_lr(self):
        return K.get_value(self.lr)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.

        # Returns
            A list of numpy arrays.
        """
        return K.batch_get_value(self.weights)

    def get_config(self):
        config = {}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def lr(self):
        # Legacy support.
        return self.learning_rate


class PAS(Optimizer):
    """Soft Passive-Agressive online learning by subgradient techniques optimizer.

    # Arguments
        trainable_weights_shapes: shapes of the model weights
        lr: float >= 0. Learning rate.
        c: float. Importance of the gradients when updating weights.
    """

    def __init__(self, trainable_weights_shapes, lr=0.01, c=1.0, **kwargs):
        super(PAS, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr, name='lr')
        self.weights = [K.zeros(shape, name=name) for name, shape in
                        trainable_weights_shapes]
        self.c = K.variable(c, name='c')
        self.loss_value = K.variable(0.0, name='loss_value')

    def set_weights(self, weights):
        for sw, w in zip(self.weights, weights):
            sw.set_value(w.eval())

    def get_weights(self):
        return self.weights

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        lr = self.lr
        C = self.c
        weights_init = self.get_weights()
        l = self.loss_value

        for wk, g, wt in zip(params, grads, weights_init):

            fd = wk - wt + l * C * g
            new_wk = wk - lr * fd

            # Apply constraints.
            if getattr(wk, 'constraint', None) is not None:
                new_wk = wk.constraint(new_wk)

            self.updates.append(K.update(wk, new_wk))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        lr = self.lr
        C = self.c
        weights_init = self.get_weights()
        l = self.loss_value

        for wk, g, lmul, wt in zip(params, grads, learning_rate_multipliers,
                                   weights_init):

            fd = wk - wt + l * C * g
            new_wk = wk - lr * lmul * fd

            # Apply constraints.
            if getattr(wk, 'constraint', None) is not None:
                new_wk = wk.constraint(new_wk)

            self.updates.append(K.update(wk, new_wk))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)), 'C': float(K.get_value(self.c))}
        base_config = super(PAS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PAS2(PAS):
    """Passive-Agressive online learning by projected subgradient techniques optimizer.
        Alternative 1
    # Arguments
        trainable_weights_shapes: shapes of the model weights
        lr: float >= 0. Learning rate.
        c: float. Weight given to projection operator.
    """

    def __init__(self, trainable_weights_shapes, lr=0.01, c=1.0, **kwargs):
        super(PAS2, self).__init__(trainable_weights_shapes, lr, c, **kwargs)
        self.__dict__.update(locals())
        self.c = K.variable(c)

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        lr = self.lr
        weights_init = self.get_weights()
        l = self.loss_value
        C = self.c

        for wk, g, wt in zip(params, grads, weights_init):

            fd = g + C * (wk - wt)
            new_wk = wk - lr * fd
            # Apply constraints.
            if getattr(wk, 'constraint', None) is not None:
                new_wk = wk.constraint(new_wk)

            self.updates.append(K.update(wk, new_wk))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        lr = self.lr
        weights_init = self.get_weights()
        l = self.loss_value
        C = self.c

        for wk, g, lmul, wt in zip(params, grads, learning_rate_multipliers,
                                   weights_init):

            fd = g + C * (wk - wt)
            new_wk = wk - lr * lmul * fd
            # Apply constraints.
            if getattr(wk, 'constraint', None) is not None:
                new_wk = wk.constraint(new_wk)

            self.updates.append(K.update(wk, new_wk))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)), 'B': float(K.get_value(self.b))}
        base_config = super(PAS2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PPAS(PAS):
    '''Passive-Agressive online learning by projected subgradient techniques optimizer.

    # Arguments
        trainable_weights_shapes: shapes of the model weights
        lr: float >= 0. Learning rate.
        c: float. Weight given to projection operator.
    '''

    def __init__(self, trainable_weights_shapes, lr=0.01, c=1.0, **kwargs):
        super(PPAS, self).__init__(trainable_weights_shapes, lr, c, **kwargs)
        self.__dict__.update(locals())
        self.b = K.variable(c)

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        lr = self.lr
        weights_init = self.get_weights()
        l = self.loss_value
        b = self.b

        for wk, g, wt in zip(params, grads, weights_init):
            new_wk = wk - lr * l * g
            p_new_wk = ((new_wk - wt) / (K.epsilon() + K.sqrt(K.sum(K.square(new_wk - wt))))) * b + wt
            # Apply constraints.
            if getattr(new_wk, 'constraint', None) is not None:
                p_new_wk = wk.constraint(p_new_wk)

            self.updates.append(K.update(wk, p_new_wk))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        lr = self.lr
        weights_init = self.get_weights()
        l = self.loss_value
        b = self.b

        for wk, g, lmul, wt in zip(params, grads, learning_rate_multipliers, weights_init):
            new_wk = wk - lr * lmul * l * g
            p_new_wk = ((new_wk - wt) / (K.epsilon() + K.sqrt(K.sum(K.square(new_wk - wt))))) * b + wt
            # Apply constraints.
            if getattr(new_wk, 'constraint', None) is not None:
                p_new_wk = wk.constraint(p_new_wk)

            self.updates.append(K.update(wk, p_new_wk))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)), 'B': float(K.get_value(self.b))}
        base_config = super(PPAS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QHSGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, Nesterov momentum,
    and quasi-hyperbolic momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        quasi_hyperbolic_momentum: Quasi hyperbolic momentum parameter
        decay: float >= 0. Learning rate decay over each update.
        dampening: boolean. Whether to apply dampening or not.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., quasi_hyperbolic_momentum=0., decay=0.,
                 dampening=0., nesterov=False, **kwargs):
        super(QHSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.dampening = K.variable(dampening, name='dampening')
            self.momentum = K.variable(momentum, name='momentum')
            self.quasi_hyperbolic_momentum = K.variable(quasi_hyperbolic_momentum,
                                                        name='quasi_hyperbolic_momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        warnings.warn(UserWarning('The QHSGD Optimizer is still somewhat untested'))
        if nesterov:
            assert momentum > 0. and dampening == 0., "Nesterov momentum requires a momentum and zero dampening"

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m + (1 - self.dampening) * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                raise NotImplementedError("NAG still unimplemented")
                new_p = p - self.momentum * lr * v - lr * g
            else:
                new_p = p - lr * ((1 - self.quasi_hyperbolic_momentum) * g + self.quasi_hyperbolic_momentum * v)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, lmul, m in zip(params, grads, learning_rate_multipliers, moments):
            v = self.momentum * m + (1 - self.dampening) * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p - self.momentum * lr * lmul * v - (lr * lmul) * g
            else:
                new_p = p - lr * lmul * ((
                                         1 - self.quasi_hyperbolic_momentum) * g + self.quasi_hyperbolic_momentum * v)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'quasi_hyperbolic_momentum': float(
                      K.get_value(self.quasi_hyperbolic_momentum)),
                  'dampening': float(K.get_value(self.dampening)),
                  'nesterov': self.nesterov}
        base_config = super(QHSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SGDHD(Optimizer):
    """Stochastic gradient descent optimizer with hypergradient descent.
    See https://openreview.net/forum?id=BkrsAzWAb

    Code adapted from the original PyTorch repo: https://github.com/gbaydin/hypergradient-descent/blob/master/sgd_hd.py

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        hypergrad_lr: (float, optional) hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        Online Learning Rate Adaptation with Hypergradient Descent (https://openreview.net/forum?id=BkrsAzWAb)
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, hypergrad_lr=1e-4, **kwargs):
        super(SGDHD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.hypergrad_lr = K.variable(hypergrad_lr, name='hypergrad_lr')
        self.initial_decay = decay
        self.nesterov = nesterov
        warnings.warn(UserWarning('The SGDHD Optimizer is still somewhat untested'))

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]

        # Initial lr moments (\nabla_{\alpha} u_0 in the HG paper)
        lr_moments = [K.zeros(shape) for shape in shapes]
        # Initial learning rate (\alpha_0)
        learning_rates = [K.variable(K.ones(shape) * lr) for shape in shapes]

        self.weights = [self.iterations] + moments
        for p, g, m, lr_m, alpha in zip(params, grads, moments, lr_moments, learning_rates):

            # Compute hypergradient
            hypergrad = g * lr_m  # In the paper: h_t := g_t \nabla_{\alpha} u_{t-1}

            # Update learning rate (\alpha_t := \alpha_{t-1} - beta * h_t)
            new_alpha = alpha - self.hypergrad_lr * hypergrad
            self.updates.append(K.update(alpha, new_alpha))

            # Compute momentum (v = \mu *v_{t-1} + g_t)
            v = self.momentum * m + g  # Velocity
            self.updates.append(K.update(m, v))  # Parameter update

            # Parameter update (-alpha_t * g_t)
            u = -new_alpha * (g + self.momentum * v)
            new_p = p + u

            # Parameter update (u_t := -g -\mu * v)
            new_lr_m = -g - self.momentum * v
            self.updates.append(K.update(lr_m, new_lr_m))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            # Apply parameter update
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]

        # Hypergrads
        lr_moments = [K.variable(K.ones(shape) * self.lr) for shape in shapes]  # Initial lr moments (alpha_0 in the HG paper)
        lr_updates = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + moments
        for p, g, lmul, m, lr_m, alpha in zip(params, grads,
                                              learning_rate_multipliers,
                                              moments, lr_moments, lr_updates):

            # Compute hypergradient
            hypergrad = g * lr_m  # h_t in the paper

            # Update hyper_lr
            new_alpha = alpha - self.hypergrad_lr * hypergrad
            self.updates.append(K.update(alpha, new_alpha))

            v = self.momentum * m - lr * lmul * g  # velocity
            self.updates.append(K.update(m, v))  # Parameter update

            new_lr_m = -g  # Parameter update (\delta_\alpha u_t
            self.updates.append(K.update(lr_m, new_lr_m))

            if self.nesterov:
                new_p = p + self.momentum * v - (lr * lmul) * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'hypergrad_lr': float(K.get_value(self.hypergrad_lr))}
        base_config = super(SGDHD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QHSGDHD(Optimizer):
    """Quasi-hyperbolic momentum stochastic gradient descent optimizer with hypergradient descent.
    See https://openreview.net/forum?id=BkrsAzWAb, https://arxiv.org/abs/1810.06801

    Consists in the SGD-HD optimizer with the momentum parametrization from https://arxiv.org/abs/1810.06801

    # Arguments
        lr: float >= 0. Initial learning rate.
        momentum: float >= 0. Momentum parameter. Accelerates SGD
        in the relevant direction and dampens oscillations. \beta parameter in  https://arxiv.org/abs/1810.06801
        quasi_hyperbolic_momentum: float >= 0. Quasi-hyperbolic momentum parameter. \mu in https://arxiv.org/abs/1810.06801
        dampening: float >=0. = dampening for momentum
        decay: learning rate decay rate.
        nesterov: boolean. Whether to apply Nesterov momentum.
        hypergrad_lr: (float >=0., optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        Online Learning Rate Adaptation with Hypergradient Descent (https://openreview.net/forum?id=BkrsAzWAb)
    """

    def __init__(self, lr=0.01, momentum=0., quasi_hyperbolic_momentum=0.,
                 dampening=0.,
                 decay=0., nesterov=False, hypergrad_lr=1e-3, **kwargs):
        super(QHSGDHD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.quasi_hyperbolic_momentum = K.variable(quasi_hyperbolic_momentum,
                                                        name='quasi_hyperbolic_momentum')
            self.dampening = K.variable(dampening, name='dampening')
            self.decay = K.variable(decay, name='decay')
            self.hypergrad_lr = K.variable(hypergrad_lr, name='hypergrad_lr')
        self.initial_decay = decay
        self.nesterov = nesterov
        warnings.warn(
            UserWarning('The QHSGDHD Optimizer is still somewhat untested'))

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]

        # Initial lr moments (\nabla_{\alpha} u_0 in the HG paper)
        lr_moments = [K.zeros(shape) for shape in shapes]
        # Initial learning rate (\alpha_0)
        learning_rates = [K.variable(K.ones(shape) * lr) for shape in shapes]

        self.weights = [self.iterations] + moments
        for p, g, m, lr_m, alpha in zip(params, grads, moments, lr_moments, learning_rates):

            # Compute hypergradient
            hypergrad = g * lr_m  # In the paper: h_t := g_t \nabla_{\alpha} u_{t-1}

            # Update learning rate (\alpha_t := \alpha_{t-1} - beta * h_t)
            new_alpha = alpha - self.hypergrad_lr * hypergrad
            self.updates.append(K.update(alpha, new_alpha))

            # Compute momentum (v = \mu *v_{t-1} + (1 - \mu) *  g_t)
            v = self.momentum * m + (1 - self.dampening) * g  # Velocity
            self.updates.append(K.update(m, v))  # Parameter update

            # Parameter update (\delta_\alpha u_t)
            u = -new_alpha * ((1 - self.quasi_hyperbolic_momentum) * g + self.quasi_hyperbolic_momentum * v)
            new_p = p + u

            # Parameter update (u_t := -g -\mu * v)
            new_lr_m = - ((1 - self.quasi_hyperbolic_momentum) * g + self.quasi_hyperbolic_momentum * v)
            self.updates.append(K.update(lr_m, new_lr_m))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            # Apply parameter update
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]

        # Hypergrads
        lr_moments = [K.variable(K.ones(shape) * self.lr) for shape in shapes]  # Initial lr moments (alpha_0 in the HG paper)
        lr_updates = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + moments
        for p, g, lmul, m, u, alpha in zip(params, grads, learning_rate_multipliers,
                                           moments, lr_moments, lr_updates):

            # Compute hypergradient
            hypergrad = g * u  # h_t in the paper
            # Update hyper_lr
            new_alpha = alpha - self.hypergrad_lr * hypergrad
            self.updates.append(K.update(alpha, new_alpha))

            # Compute momentum
            v = self.momentum * m - (1 - self.dampening) * g  # velocity
            self.updates.append(K.update(m, v))  # Parameter update

            # Parameter update (\delta_\alpha u_t)
            new_u = - ((
                       1 - self.quasi_hyperbolic_momentum) * g + self.quasi_hyperbolic_momentum * v)
            self.updates.append(K.update(u, new_u))

            if self.nesterov:
                raise NotImplementedError("NAG still unimplemented")
                new_p = p + self.momentum * v - (lr * lmul) * g
            else:
                new_p = p + new_alpha * new_u

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'quasi_hyperbolic_momentum': float(
                      K.get_value(self.quasi_hyperbolic_momentum)),
                  'dampening': float(K.get_value(self.dampening)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'hypergrad_lr': float(K.get_value(self.hypergrad_lr))}
        base_config = super(QHSGDHD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, learning_rate=0.01, momentum=0.,
                 nesterov=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        self.initial_decay = kwargs.pop('decay', 0.0)
        super(SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(self.initial_decay, name='decay')
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape, name='moment_' + str(i))
                   for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, lmul, m in zip(params, grads, learning_rate_multipliers, moments):
            v = self.momentum * m - lr * lmul * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - (lr * lmul) * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RMSprop(Optimizer):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    # Arguments
        learning_rate: float >= 0. Learning rate.
        rho: float >= 0.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude
           ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, learning_rate=0.001, rho=0.9, **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RMSprop, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p),
                        dtype=K.dtype(p),
                        name='accumulator_' + str(i))
                        for (i, p) in enumerate(params)]
        self.weights = [self.iterations] + accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super(RMSprop, self).set_weights(weights)

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adagrad(Optimizer):
    """Adagrad optimizer.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the learning rate.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Initial learning rate.

    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, learning_rate=0.01, **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Adagrad, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):

        # The comments are taken from Fig. 1 from the AdaGrad paper.

        # 0. Suffer lox f_1(x_t) -> loss
        # 1. Recieve subgradient g_t
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape, name='accumulator_' + str(i))
                        for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):

        # The comments are taken from Fig. 1 from the AdaGrad paper.

        # 0. Suffer lox f_1(x_t) -> loss
        # 1. Recieve subgradient g_t
        grads = self.get_gradients(loss, params)

        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, lmul in zip(params, grads, accumulators,
                                 learning_rate_multipliers):
            new_a = a + K.square(g)  # update accumulator G_t
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * lmul * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super(Adagrad, self).set_weights(weights)

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adadelta(Optimizer):
    """Adadelta optimizer.

    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate. In this version, initial learning rate and decay factor can
    be set, as in most other Keras optimizers.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Initial learning rate, defaults to 1.
            It is recommended to leave it at the default value.
        rho: float >= 0. Adadelta decay factor, corresponding to fraction of
            gradient to keep at each time step.

    # References
        - [Adadelta - an adaptive learning rate method](
           https://arxiv.org/abs/1212.5701)
    """

    def __init__(self, learning_rate=1.0, rho=0.95, **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Adadelta, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.rho = rho

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape, name='accumulator_' + str(i))
                        for (i, shape) in enumerate(shapes)]
        delta_accumulators = [K.zeros(shape, name='delta_accumulator_' + str(i))
                              for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + accumulators + delta_accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)
            new_p = p - lr * update

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append(K.update(d_a, new_d_a))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        delta_accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, d_a, lmul in zip(params, grads, accumulators,
                                      delta_accumulators, learning_rate_multipliers):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)
            new_p = p - lr * lmul * update

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append(K.update(d_a, new_d_a))
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super(Adadelta, self).set_weights(weights)

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'rho': self.rho,
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adadelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adam(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".

    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 amsgrad=False, **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Adam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')
        self.amsgrad = amsgrad

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='v_' + str(i))
              for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p),
                     dtype=K.dtype(p),
                     name='vhat_' + str(i))
                     for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i))
                     for i in range(len(params))]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, lmul in zip(params, grads, ms, vs, vhats,
                                          learning_rate_multipliers):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t * lmul / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t * lmul / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdamHD(Optimizer):
    """Adam optimizer with hypergradient descent.
    See https://openreview.net/forum?id=BkrsAzWAb

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        hypergrad_lr: (float, optional) hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        Online Learning Rate Adaptation with Hypergradient Descent (https://openreview.net/forum?id=BkrsAzWAb)
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".

    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, hypergrad_lr=1e-3,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(AdamHD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.hypergrad_lr = K.variable(hypergrad_lr, name='hypergrad_lr')

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        bias_corrections = (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        # Initial lr moments (\nabla_{\alpha} u_0 in the HG paper)
        lr_moments = [K.zeros(shape) for shape in shapes]
        # Initial learning rate (\alpha_0)
        learning_rates = [K.variable(K.ones(shape) * lr) for shape in shapes]

        for p, g, m, v, vhat, lr_m, alpha in zip(params, grads, ms, vs, vhats,
                                                 lr_moments, learning_rates):

            # Compute hypergradient
            hypergrad = g * lr_m  # In the paper: h_t := g_t \nabla_{\alpha} u_{t-1}

            # Update learning rate (\alpha_t := \alpha_{t-1} - beta * h_t)
            new_alpha = alpha - self.hypergrad_lr * hypergrad
            self.updates.append(K.update(alpha, new_alpha))

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_alpha * bias_corrections * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_alpha * bias_corrections * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

            # Parameter update (u_t := -\frac{\hat{m_t} /(\sqrt{\hat{v_t}} + epsilon}
            if self.amsgrad:
                new_lr_m = -bias_corrections * m_t / (K.sqrt(vhat_t) + self.epsilon)
            else:
                new_lr_m = -bias_corrections * m_t / (K.sqrt(v_t) + self.epsilon)
            self.updates.append(K.update(lr_m, new_lr_m))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'hypergrad_lr': float(K.get_value(self.hypergrad_lr)),
                  'amsgrad': self.amsgrad}
        base_config = super(AdamHD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdamAccumulate(Optimizer):
    """Adam optimizer updating parameters only after a number of iterations.
    Implementation made by: https://github.com/keras-team/keras/issues/3556#issuecomment-417317758
    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        accum_iters: int > 0. Number of iteration between gradient updates.
            If 1, the optimizer falls to regular Adam.

    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.,
                 amsgrad=False, accum_iters=20, **kwargs):
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, dtype='int64')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, gg in zip(params, grads, ms, vs, vhats, gs):

            update_gradients = K.cast(K.equal(self.iterations % self.accum_iters, 0),
                                      K.floatx())
            gg_t = (1 - update_gradients) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + update_gradients * g) / K.cast(self.accum_iters, K.floatx())
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((gg + update_gradients * g) / K.cast(self.accum_iters, K.floatx()))
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(
                (m, update_gradients * m_t + (1 - update_gradients) * m))
            self.updates.append(
                (v, update_gradients * v_t + (1 - update_gradients) * v))
            self.updates.append((gg, gg_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, gg in zip(params, grads, ms, vs, vhats, gs):

            update_gradients = K.cast(K.equal(self.iterations % self.accum_iters, 0),
                                      K.floatx())
            gg_t = (1 - update_gradients) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + update_gradients * g) / K.cast(self.accum_iters, K.floatx())
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(
                (gg + update_gradients * g) / K.cast(self.accum_iters, K.floatx()))
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(
                (m, update_gradients * m_t + (1 - update_gradients) * m))
            self.updates.append(
                (v, update_gradients * v_t + (1 - update_gradients) * v))
            self.updates.append((gg, gg_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'accum_iters': float(K.get_value(self.accum_iters))}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adamax(Optimizer):
    """Adamax optimizer from Adam paper's Section 7.

    It is a variant of Adam based on the infinity norm.
    Default parameters follow those provided in the paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.

    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Adamax, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr / (1. - K.pow(self.beta_1, t))

        shapes = [K.int_shape(p) for p in params]
        # zero init of 1st moment
        ms = [K.zeros(shape, name='m_' + str(i))
              for (i, shape) in enumerate(shapes)]
        # zero init of exponentially weighted infinity norm
        us = [K.zeros(shape, name='u_' + str(i))
              for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + ms + us

        for p, g, m, u in zip(params, grads, ms, us):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = K.maximum(self.beta_2 * u, K.abs(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(u, u_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr / (1. - K.pow(self.beta_1, t))

        shapes = [K.int_shape(p) for p in params]
        # zero init of 1st moment
        ms = [K.zeros(shape) for shape in shapes]
        # zero init of exponentially weighted infinity norm
        us = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + us

        for p, g, m, u, lmul in zip(params, grads, ms, us,
                                    learning_rate_multipliers):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = K.maximum(self.beta_2 * u, K.abs(g))
            p_t = p - (lr_t * lmul) * m_t / (u_t + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(u, u_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adamax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Nadam(Optimizer):
    """Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](
           http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, **kwargs):
        self.schedule_decay = kwargs.pop('schedule_decay', 0.004)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Nadam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape, name='m_' + str(i))
              for (i, shape) in enumerate(shapes)]
        vs = [K.zeros(shape, name='v_' + str(i))
              for (i, shape) in enumerate(shapes)]

        self.weights = [self.iterations, self.m_schedule] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = (p - self.learning_rate * m_t_bar / (K.sqrt(v_t_prime) +
                   self.epsilon))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include m_schedule at head of the weight list. Set
        # m_schedule to 1.
        if len(params) == len(weights) + 1:
            weights = [weights[0]] + [np.array(1.)] + weights[1:]
        super(Nadam, self).set_weights(weights)

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay}
        base_config = super(Nadam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TFOptimizer(Optimizer):
    """Wrapper class for native TensorFlow optimizers.

    # Arguments
        optimizer: Selected optimizer
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        if isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
            return self.optimizer.get_updates(loss, params)
        else:
            grads = self.optimizer.compute_gradients(loss, var_list=params)
        self.updates = [K.update_add(self.iterations, 1)]
        opt_update = self.optimizer.apply_gradients(
            grads, global_step=self.iterations)
        self.updates.append(opt_update)
        return self.updates

    def get_updates_with_lr_multipliers(self, loss, params,
                                        learning_rate_multipliers):
        grads = self.optimizer.compute_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        opt_update = self.optimizer.apply_gradients(
            grads, global_step=self.iterations)
        self.updates.append(opt_update)
        return self.updates

    @property
    def weights(self):
        if isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
            return self.optimizer.weights
        raise NotImplementedError

    def get_config(self):
        if isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
            return self.optimizer.get_config
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        if tf.__version__.startswith('1.'):
            raise NotImplementedError
        return cls(**config)


# Aliases.

sgd = SGD
sgdhd = SGDHD
qhsgdhd = QHSGDHD
qhsgd = QHSGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamhd = AdamHD
adamaccumulate = AdamAccumulate
adamax = Adamax
nadam = Nadam


def serialize(optimizer):
    return serialize_keras_object(optimizer)


def deserialize(config, custom_objects=None):
    """Inverse of the `serialize` function.

    # Arguments
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping
            names (strings) to custom objects
            (classes and functions)
            to be considered during deserialization.

    # Returns
        A Keras Optimizer instance.
    """
    all_classes = {
        'sgd': SGD,
        'qhsgd': QHSGD,
        'sgdhd': SGDHD,
        'qhsgdhd': QHSGDHD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamhd': AdamHD,
        'adamax': Adamax,
        'nadam': Nadam,
        'tfoptimizer': TFOptimizer,
    }
    # Make deserialization case-insensitive for built-in optimizers.
    if config['class_name'].lower() in all_classes:
        config['class_name'] = config['class_name'].lower()
    return deserialize_keras_object(config,
                                    module_objects=all_classes,
                                    custom_objects=custom_objects,
                                    printable_module_name='optimizer')


def get(identifier):
    """Retrieves a Keras Optimizer instance.

    # Arguments
        identifier: Optimizer identifier, one of
            - String: name of an optimizer
            - Dictionary: configuration dictionary.
            - Keras Optimizer instance (it will be returned unchanged).
            - TensorFlow Optimizer instance
                (it will be wrapped as a Keras Optimizer).

    # Returns
        A Keras Optimizer instance.

    # Raises
        ValueError: If `identifier` cannot be interpreted.
    """
    if K.backend() == 'tensorflow':
        # Wrap TF optimizer instances
        if tf.__version__.startswith('1.'):
            try:
                TFOpt = tf.compat.v1.train.Optimizer
            except AttributeError:
                TFOpt = tf.train.Optimizer
            if isinstance(identifier, TFOpt):
                return TFOptimizer(identifier)
        elif isinstance(identifier, tf.keras.optimizers.Optimizer):
            return TFOptimizer(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    if isinstance(identifier, Optimizer):
        return identifier
    else:
        raise ValueError('Could not interpret optimizer identifier: ' +
                         str(identifier))
