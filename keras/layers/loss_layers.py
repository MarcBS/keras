# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import backend as K

from ..engine import Layer


# Custom loss layer
class CustomMultiLossLayer(Layer):
    """CustomMultiLossLayer. Adds losses to the model.
        TODO: Write docs.

    # Arguments
        nb_outputs: Number of outputs
        loss_functions: List of loss functions to add.
    """

    def __init__(self, nb_outputs=2, loss_functions=[], **kwargs):
        self.nb_outputs = nb_outputs
        self.loss_functions = loss_functions
        self.is_placeholder = True
        assert len(loss_functions) == nb_outputs
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer='zeros', trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var, loss_func in zip(ys_true, ys_pred, self.log_vars, self.loss_functions):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * loss_func(y_true, y_pred) + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

    def get_config(self):
        config = {
            'nb_outputs': self.nb_outputs,
            'loss_functions': self.loss_functions,
        }
        base_config = super(Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Losses uncertainty loss
def deactivate_loss():
    def loss(y_true, y_pred):
        cost = K.variable(0)
        return cost

    return loss


def uncertainty_categorical_crossentropy(auto_weight=True, norm=False):
    """uncertainty_categorical_crossentropy.
        TODO: Write docs.

    # Arguments
        auto_weight: ...
        norm: ..

    # Returns
        loss: ...
    """
    def loss(y_true, y_pred):
        # categorical cross entropy loss (L(W) single task labels)
        y_pred_top = y_pred[:, :-1]

        if auto_weight:
            logsigma2 = y_pred[:, -1]
        else:
            logsigma2 = K.variable(0.0)

        cost = K.categorical_crossentropy(y_true, y_pred_top)
        # L(W,s)
        cost = K.exp(-logsigma2) * cost + logsigma2
        # Ln(W,s)
        if norm:
            if K.backend() == 'tensorflow':
                max_loss = -K.log(K.exp(-1.0) / (K.exp(1.0) * (float(K.int_shape(y_pred_top)[-1]))))
            elif K.backend() == 'theano':
                max_loss = -K.log(K.exp(-1.0) / (K.exp(1.0) * (K.shape(y_pred_top)[-1])))
            else:
                raise NotImplementedError("Loss function not implemented for the chosen backend.")
            cost = cost / max_loss
        cost = K.mean(cost)
        return cost

    return loss


def uncertainty_binary_crossentropy(auto_weight=True, norm=False):
    """uncertainty_binary_crossentropy.
        TODO: Write docs.

    # Arguments
        auto_weight: ...
        norm: ..

    # Returns
        loss: ...
    """
    def loss(y_true, y_pred):
        # categorical cross entropy loss (L(W) single task labels)
        y_pred_top = y_pred[:, :-1]

        if auto_weight:
            logsigma2 = y_pred[:, -1]
        else:
            logsigma2 = K.variable(0.0)

        cost = K.sum(K.binary_crossentropy(y_true, y_pred_top), axis=-1)
        # L(W,s)
        if K.backend() == 'tensorflow':
            nlabels = K.cast_to_floatx(K.int_shape(y_true)[-1])
        elif K.backend() == 'theano':
            nlabels = K.shape(y_true)[-1]
        else:
            raise NotImplementedError("Loss function not implemented for the chosen backend.")
        cost = K.exp(-logsigma2) * cost + nlabels * logsigma2
        # Ln(W,s)
        if norm:
            max_loss = nlabels
            cost = cost / max_loss
        cost = K.mean(cost)
        return cost

    return loss
