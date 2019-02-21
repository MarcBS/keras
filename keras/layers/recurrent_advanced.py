# -*- coding: utf-8 -*-
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from .. import backend as K
from .. import activations
from .. import initializers
from .. import regularizers
from .. import constraints

from ..engine.base_layer import Layer
from ..engine.base_layer import InputSpec
from ..utils.generic_utils import has_arg
from ..utils.generic_utils import to_list
from .recurrent import RNN

# Legacy support.
from ..legacy.layers import Recurrent
from ..legacy import interfaces


def compute_attention(h_tm1, pctx_, context, att_dp_mask, attention_recurrent_kernel,
                      attention_context_wa, bias_ca, mask_context, attention_mode='add'):
    """Computes an attended vector over an input sequence of vectors (context).

    The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over the sequence of inputs 'x_1^I':
            phi(x_1^I, t) = ∑_i alpha_i(t) * x_i,
        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that accomplishes the following condition:
            ∑_i alpha_i = 1
        and is dynamically adapted at each timestep w.r.t. the following formula:
            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}
        where each 'e_i' at time 't' is calculated as:
            e_i(t) = score(h_tm1, x_i)

        score is a function that assigns a weight depending on how well h_tm1 and x_i match.
        The following scoring functions are implemented:
            - 'add'/'bahdanau':
               e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h_tm1 +  ba ),
            - 'dot'/'luong':
               e_i(t) = h_tm1' · x_i # Requires the dimensions to be the same
            - 'scale-dot':
               e_i(t) = (h_tm1' · x_i) / \sqrt(|x_i|) # Requires the dimensions to be the same

    # Arguments
        h_tm1: Last decoder state.
        pctx_: Projected context (i.e. context * Ua + ba).
        context: Original context.
        att_dp_mask: Dropout for the attention MLP.
        attention_recurrent_kernel:  attention MLP weights.
        attention_context_wa:  attention MLP weights.
        bias_ca:  attention MLP bias.
        mask_context: mask of the context.
        attention_mode: 'add', 'dot' or function that accepts as arguments: `h_tm1, pctx_, context, att_dp_mask, attention_recurrent_kernel, attention_context_wa, bias_ca, mask_context`
        and should return the scores `e` for the input annotations.

    # Returns
        ctx_: attended representation of the input.
        alphas: weights computed by the attention mechanism.

    # Raises
        NotImplementedError: If the attention_mode specified is not implemented.

    # References
        - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
        - [Effective Approaches to Attention-based Neural Machine Translation](http://www.aclweb.org/anthology/D15-1166)
    """
    p_state_ = K.dot_product(h_tm1 * att_dp_mask[0], attention_recurrent_kernel)

    if attention_mode == 'add' or attention_mode == 'bahdanau':
        pctx_ = K.tanh(pctx_ + p_state_[:, None, :])
        e = K.dot_product(pctx_, attention_context_wa) + bias_ca

    elif attention_mode == 'dot' or attention_mode == 'luong':
        pctx_ = K.batch_dot(p_state_[:, :, None], pctx_, axes=[1, 2])
        e = K.squeeze(pctx_, 1)
    elif attention_mode == 'scaled-dot':
        pctx_ = K.batch_dot(p_state_[:, :, None], pctx_, axes=[1, 2]) / K.sqrt(K.cast(K.shape(pctx_)[-1], K.floatx()))
        e = K.squeeze(pctx_, 1)
    elif hasattr(attention_mode, '__call__'):
        e = attention_mode(h_tm1, pctx_, context, att_dp_mask, attention_recurrent_kernel,
                           attention_context_wa, bias_ca, mask_context)
    else:
        raise NotImplementedError('The attention mode ' + attention_mode + ' is not implemented.')

    if mask_context is not None and K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
        e = K.cast(mask_context, K.dtype(e)) * e
    alphas = K.softmax(K.reshape(e, [K.shape(e)[0], K.shape(e)[1]]))

    # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
    ctx_ = K.sum(context * alphas[:, :, None], axis=1)

    return ctx_, alphas


class GRUCond(Recurrent):
    """Gated Recurrent Unit - Cho et al. 2014. with the previously generated word fed to the current timestep.
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The input context  (shape: (batch_size, context_size))

    # Arguments
        units: Positive integer, dimensionality of the output space.
        return_states: Whether it should return the internal RNN states.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        num_inputs: Number of inputs of the layer.
        static_ctx: If static_ctx, it should have 2 dimensions and it will
                    be fed to each timestep of the RNN. Otherwise, it should
                    have 3 dimensions and should have the same number of
                    timesteps than the input.
    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 return_states=False,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 num_inputs=3,
                 static_ctx=False,
                 **kwargs):

        super(GRUCond, self).__init__(**kwargs)

        self.return_states = return_states

        # Main parameters
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        if static_ctx:
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        else:
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) == 2 or len(input_shape) == 3, 'You should pass two inputs to GRUCond ' \
                                                               '(context and previous_embedded_words) and ' \
                                                               'one optional inputs (init_state). ' \
                                                               'It currently has %d inputs' % len(input_shape)

        self.input_dim = input_shape[0][2]
        if self.input_spec[1].ndim == 3:
            self.context_dim = input_shape[1][2]
            self.static_ctx = False
            assert input_shape[1][1] == input_shape[0][1], 'When using a 3D ctx in GRUCond, it has to have the same ' \
                                                           'number of timesteps (dimension 1) as the input. Currently,' \
                                                           'the number of input timesteps is: ' \
                                                           + str(input_shape[0][1]) + \
                                                           ', while the number of ctx timesteps is ' \
                                                           + str(input_shape[1][1]) + ' (complete shapes: ' \
                                                           + str(input_shape[0]) + ', ' + str(input_shape[1]) + ')'
        else:
            self.context_dim = input_shape[1][1]
            self.static_ctx = True

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None]  # [h, c]

        self.kernel = self.add_weight(shape=(self.context_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units))]

    def preprocess_input(self, inputs, training=None):

        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(3)]
            preprocessed_input = K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            preprocessed_input = K.dot(inputs, self.conditional_kernel)

        if self.static_ctx:
            return preprocessed_input

        # Not Static ctx
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs, ones,
                                        training=training) for _ in range(3)]
            preprocessed_context = K.dot(self.context * dp_mask[0][:, None, :], self.kernel)
        else:
            preprocessed_context = K.dot(self.context, self.kernel)
        return preprocessed_input + preprocessed_context

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_states:
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out = [main_out, states_dim]
        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context = inputs[1]
        if self.num_inputs == 2:  # input: [state_below, context]
            self.init_state = None
        elif self.num_inputs == 3:  # input: [state_below, context, init_generic]
            self.init_state = inputs[2]
        elif self.num_inputs == 4:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[2]
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_sequences:
            ret = K.cast(mask[0], K.floatx())
        else:
            ret = None
        if self.return_states:
            ret = [ret, None]
        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        rec_dp_mask = states[1]  # Dropout U (recurrent)
        matrix_x = x
        if self.use_bias:
            matrix_x = K.bias_add(matrix_x, self.bias)

        if self.static_ctx:
            dp_mask = states[3]  # Dropout W
            context = states[4]
            mask_context = states[5]  # Context mask
            if K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
                context = K.cast(mask_context[:, :, None], K.dtype(context)) * context
            matrix_x += K.dot(context * dp_mask[0], self.kernel)

        matrix_inner = K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel[:, :2 * self.units])
        x_z = matrix_x[:, :self.units]
        x_r = matrix_x[:, self.units: 2 * self.units]
        inner_z = matrix_inner[:, :self.units]
        inner_r = matrix_inner[:, self.units: 2 * self.units]

        z = self.recurrent_activation(x_z + inner_z)
        r = self.recurrent_activation(x_r + inner_r)

        x_h = matrix_x[:, 2 * self.units:]
        inner_h = K.dot(r * h_tm1 * rec_dp_mask[0], self.recurrent_kernel[:, 2 * self.units:])
        hh = self.activation(x_h + inner_h)
        h = z * h_tm1 + (1 - z) * hh

        return h, [h]

    def get_constants(self, inputs, mask_context, training=None):
        constants = []
        # States[2] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        # States[3] - Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(3)]
        else:
            dp_mask = [K.cast_to_floatx(1.) for _ in range(3)]

        if self.static_ctx:
            constants.append(dp_mask)

        # States[4] - context
        constants.append(self.context)

        # States[5] - mask_context
        if mask_context is None:
            mask_context = K.not_equal(K.sum(self.context, axis=2), self.mask_value)
            mask_context = K.cast(mask_context, K.floatx())
        constants.append(mask_context)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        else:
            initial_state = self.init_state
        initial_states = [initial_state]

        return initial_states

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'return_states': self.return_states,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'conditional_initializer': initializers.serialize(self.conditional_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'conditional_regularizer': regularizers.serialize(self.conditional_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'conditional_constraint': constraints.serialize(self.conditional_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'conditional_dropout': self.conditional_dropout,
                  'mask_value': self.mask_value,
                  'static_ctx': self.static_ctx,
                  'num_inputs': self.num_inputs
                  }
        base_config = super(GRUCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttGRU(Recurrent):
    """Gated Recurrent Unit with Attention
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        return_states: Whether it should return the internal RNN states.
        attention_mode: 'add', 'dot' or custom function.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            Fraction of the units to drop for
            the linear transformation in the attended context.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        num_inputs: Number of inputs of the layer.


    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [units, input_dim] [input_dim]

        The names of 'Ua' and 'Wa' are exchanged w.r.t. the provided reference as well as 'v' being renamed
        to 'x' for matching Keras LSTM's nomenclature.

    # References
        -   Yao L, Torabi A, Cho K, Ballas N, Pal C, Larochelle H, Courville A.
            Describing videos by exploiting temporal structure.
            InProceedings of the IEEE International Conference on Computer Vision 2015 (pp. 4507-4515).
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 att_units=0,
                 return_extra_variables=False,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 attention_recurrent_regularizer=None,
                 attention_context_regularizer=None,
                 attention_context_wa_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_context_constraint=None,
                 attention_context_wa_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 attention_dropout=0.,
                 num_inputs=3,
                 **kwargs):
        super(AttGRU, self).__init__(**kwargs)
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.att_units = units if att_units == 0 else att_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) >= 2, 'You should pass two inputs to AttGRU ' \
                                      '(previous_embedded_words and context) ' \
                                      'and two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[0][1]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None]  # [h, c, x_att]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.attention_recurrent_kernel = self.add_weight(
            shape=(self.units, self.att_units),
            name='attention_recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(
            shape=(self.input_dim, self.att_units),
            name='attention_context_kernel',
            initializer=self.attention_context_initializer,
            regularizer=self.attention_context_regularizer,
            constraint=self.attention_context_constraint)

        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(
                shape=(self.att_units,),
                name='attention_context_wa',
                initializer=self.attention_context_wa_initializer,
                regularizer=self.attention_context_wa_regularizer,
                constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.bias_ba = self.add_weight(shape=(self.att_units,),
                                       name='bias_ba',
                                       initializer=self.bias_ba_initializer,
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           name='bias_ca',
                                           initializer=self.bias_ca_initializer,
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None

        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):
        return inputs

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim]

        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        if self.num_inputs == 1:  # input: [context]
            self.init_state = None
        elif self.num_inputs == 2:  # input: [context, init_generic]
            self.init_state = inputs[1]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[1, 2])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[1], states[2]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]

        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        non_used_x_att = states[1]  # Placeholder for returning extra variables
        non_used_alphas_att = states[2]  # Placeholder for returning extra variables
        dp_mask = states[3]  # Dropout W (input)
        rec_dp_mask = states[4]  # Dropout U (recurrent)
        # Att model dropouts
        att_dp_mask = states[5]  # Dropout Wa
        pctx_ = states[6]  # Projected context (i.e. context * Ua + ba)
        context = states[7]  # Original context

        ctx_, alphas = compute_attention(h_tm1, pctx_, context, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, None,
                                         attention_mode=self.attention_mode)

        matrix_x = x + K.dot(ctx_ * dp_mask[0], self.kernel)
        if self.use_bias:
            matrix_x = K.bias_add(matrix_x, self.bias)
        matrix_inner = K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel[:, :2 * self.units])

        x_z = matrix_x[:, :self.units]
        x_r = matrix_x[:, self.units: 2 * self.units]
        recurrent_z = matrix_inner[:, :self.units]
        recurrent_r = matrix_inner[:, self.units: 2 * self.units]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        x_h = matrix_x[:, 2 * self.units:]
        recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
                            self.recurrent_kernel[:, 2 * self.units:])
        hh = self.activation(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True

        return h, [h, ctx_, alphas]

    def get_constants(self, inputs, training=None):
        constants = []
        # States[4] - Dropout_W
        if 0 < self.dropout < 1:
            # TODO: Fails?
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[5] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[6]  - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.attention_dropout < 1:
            input_dim = K.shape(inputs)[2]
            ones = K.ones_like(K.reshape(inputs[:, :, 0], (-1, K.shape(inputs)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx = K.dot(inputs * B_Ua[0], self.attention_context_kernel)
        else:
            pctx = K.dot(inputs, self.attention_context_kernel)
        if self.use_bias:
            pctx = K.bias_add(pctx, self.bias_ba)
        # States[7] - pctx_
        constants.append(pctx)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        else:
            initial_state = self.init_state
        initial_states = [initial_state]

        initial_state = K.zeros_like(self.context)  # (samples, input_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)  # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)  # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states

    def get_config(self):
        config = {'units': self.units,
                  "att_units": self.att_units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'return_extra_variables': self.return_extra_variables,
                  'return_states': self.return_states,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'attention_dropout': self.attention_dropout,
                  'mask_value': self.mask_value,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttGRUCond(Recurrent):
    """Gated Recurrent Unit with Attention
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        return_states: Whether it should return the internal RNN states.
        attention_mode: 'add', 'dot' or custom function.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        num_inputs: Number of inputs of the layer.


    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [units, input_dim] [input_dim]

        The names of 'Ua' and 'Wa' are exchanged w.r.t. the provided reference as well as 'v' being renamed
        to 'x' for matching Keras LSTM's nomenclature.

    # References
        -   Yao L, Torabi A, Cho K, Ballas N, Pal C, Larochelle H, Courville A.
            Describing videos by exploiting temporal structure.
            InProceedings of the IEEE International Conference on Computer Vision 2015 (pp. 4507-4515).
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 att_units=0,
                 return_extra_variables=False,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 attention_recurrent_regularizer=None,
                 attention_context_regularizer=None,
                 attention_context_wa_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_context_constraint=None,
                 attention_context_wa_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 num_inputs=3,
                 **kwargs):
        super(AttGRUCond, self).__init__(**kwargs)
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.att_units = units if att_units == 0 else att_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) >= 2, 'You should pass two inputs to AttGRUCond ' \
                                      '(previous_embedded_words and context) ' \
                                      'and two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[1][1]
        self.context_dim = input_shape[1][2]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None]  # [h, x_att]

        self.kernel = self.add_weight(shape=(self.context_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(
            shape=(self.units, self.att_units),
            name='attention_recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(
            shape=(self.context_dim, self.att_units),
            name='attention_context_kernel',
            initializer=self.attention_context_initializer,
            regularizer=self.attention_context_regularizer,
            constraint=self.attention_context_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(
                shape=(self.att_units,),
                name='attention_context_wa',
                initializer=self.attention_context_wa_initializer,
                regularizer=self.attention_context_wa_regularizer,
                constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.bias_ba = self.add_weight(shape=(self.att_units,),
                                       name='bias_ba',
                                       initializer=self.bias_ba_initializer,
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           name='bias_ca',
                                           initializer=self.bias_ca_initializer,
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None
        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):

        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(3)]
            return K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            return K.dot(inputs, self.conditional_kernel)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim]

        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context = inputs[1]
        if self.num_inputs == 2:  # input: [state_below, context]
            self.init_state = None
        elif self.num_inputs == 3:  # input: [state_below, context, init_generic]
            self.init_state = inputs[2]
        elif self.num_inputs == 4:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[2]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[1, 2])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[1], states[2]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0]]

        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        non_used_x_att = states[1]  # Placeholder for returning extra variables
        non_used_alphas_att = states[2]  # Placeholder for returning extra variables
        dp_mask = states[3]  # Dropout W (input)
        rec_dp_mask = states[4]  # Dropout U (recurrent)
        # Att model dropouts
        att_dp_mask = states[5]  # Dropout Wa
        pctx_ = states[6]  # Projected context (i.e. context * Ua + ba)
        context = states[7]  # Original context
        mask_context = states[8]  # Context mask
        if K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
            pctx_ = K.cast(mask_context[:, :, None], K.dtype(pctx_)) * pctx_
            context = K.cast(mask_context[:, :, None], K.dtype(context)) * context

        ctx_, alphas = compute_attention(h_tm1, pctx_, context, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, mask_context,
                                         attention_mode=self.attention_mode)

        matrix_x = x + K.dot(ctx_ * dp_mask[0], self.kernel)
        if self.use_bias:
            matrix_x = K.bias_add(matrix_x, self.bias)
        matrix_inner = K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel[:, :2 * self.units])

        x_z = matrix_x[:, :self.units]
        x_r = matrix_x[:, self.units: 2 * self.units]
        recurrent_z = matrix_inner[:, :self.units]
        recurrent_r = matrix_inner[:, self.units: 2 * self.units]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        x_h = matrix_x[:, 2 * self.units:]
        recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0], self.recurrent_kernel[:, 2 * self.units:])
        hh = self.activation(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, ctx_, alphas]

    def get_constants(self, inputs, mask_context, training=None):
        constants = []
        # States[4] - Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        # States[5] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        # States[6]  - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.attention_dropout < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, K.shape(self.context)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.attention_context_kernel)
        else:
            pctx = K.dot(self.context, self.attention_context_kernel)
        if self.use_bias:
            pctx = K.bias_add(pctx, self.bias_ba)
        # States[7] - pctx_
        constants.append(pctx)

        # States[8] - context
        constants.append(self.context)

        # States[9] - mask_context
        if mask_context is None:
            mask_context = K.not_equal(K.sum(self.context, axis=2), self.mask_value)
            mask_context = K.cast(mask_context, K.floatx())
        constants.append(mask_context)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        else:
            initial_state = self.init_state
        initial_states = [initial_state]

        initial_state = K.zeros_like(self.context)  # (samples, input_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)  # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)  # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states

    def get_config(self):
        config = {'units': self.units,
                  "att_units": self.att_units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'return_extra_variables': self.return_extra_variables,
                  'return_states': self.return_states,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'conditional_initializer': initializers.serialize(self.conditional_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'attention_context_initializer': initializers.serialize(self.attention_context_initializer),
                  'attention_context_wa_initializer': initializers.serialize(self.attention_context_wa_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'conditional_regularizer': regularizers.serialize(self.conditional_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'attention_context_regularizer': regularizers.serialize(self.attention_context_regularizer),
                  'attention_context_wa_regularizer': regularizers.serialize(self.attention_context_wa_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'conditional_constraint': constraints.serialize(self.conditional_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'attention_context_constraint': constraints.serialize(self.attention_context_constraint),
                  'attention_context_wa_constraint': constraints.serialize(self.attention_context_wa_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'conditional_dropout': self.conditional_dropout,
                  'attention_dropout': self.attention_dropout,
                  'mask_value': self.mask_value,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttGRUCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttConditionalGRUCond(Recurrent):
    """Conditional Gated Recurrent Unit - Cho et al. 2014. with Attention + the previously generated word fed to the current timestep.

    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        return_states: Whether it should return the internal RNN states.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        attention_mode: 'add', 'dot' or custom function.
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        num_inputs: Number of inputs of the layer.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Nematus: a Toolkit for Neural Machine Translation](http://arxiv.org/abs/1703.04357)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 att_units=0,
                 return_extra_variables=False,
                 return_states=False,
                 activation='tanh',
                 attention_mode='add',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 attention_recurrent_regularizer=None,
                 attention_context_regularizer=None,
                 attention_context_wa_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_context_constraint=None,
                 attention_context_wa_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 num_inputs=3,
                 **kwargs):
        super(AttConditionalGRUCond, self).__init__(**kwargs)
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.att_units = units if att_units == 0 else att_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.recurrent1_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias1_initializer = initializers.get(bias_initializer)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.recurrent1_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias1_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.recurrent1_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias1_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) >= 2, 'You should pass two inputs to AttConditionalGRUCond ' \
                                      '(previous_embedded_words and context) ' \
                                      'and two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[1][1]
        self.context_dim = input_shape[1][2]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None]  # [h, x_att]

        self.kernel = self.add_weight(shape=(self.context_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent1_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent1_kernel',
            initializer=self.recurrent1_initializer,
            regularizer=self.recurrent1_regularizer,
            constraint=self.recurrent1_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(
            shape=(self.units, self.att_units),
            name='attention_recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(
            shape=(self.context_dim, self.att_units),
            name='attention_context_kernel',
            initializer=self.attention_context_initializer,
            regularizer=self.attention_context_regularizer,
            constraint=self.attention_context_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(
                shape=(self.att_units,),
                name='attention_context_wa',
                initializer=self.attention_context_wa_initializer,
                regularizer=self.attention_context_wa_regularizer,
                constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.bias1 = self.add_weight(shape=(self.units * 3,),
                                         name='bias1',
                                         initializer=self.bias1_initializer,
                                         regularizer=self.bias1_regularizer,
                                         constraint=self.bias1_constraint)
        else:
            self.bias = None
            self.bias1 = None

        self.bias_ba = self.add_weight(shape=(self.att_units,),
                                       name='bias_ba',
                                       initializer=self.bias_ba_initializer,
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           name='bias_ca',
                                           initializer=self.bias_ca_initializer,
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None

        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):

        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(3)]
            return K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)

        else:
            return K.dot(inputs, self.conditional_kernel)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim]

        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context = inputs[1]
        if self.num_inputs == 2:  # input: [state_below, context]
            self.init_state = None
        elif self.num_inputs == 3:  # input: [state_below, context, init_generic]
            self.init_state = inputs[2]
        elif self.num_inputs == 4:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[2]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[1, 2])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[1], states[2]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0]]

        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        non_used_x_att = states[1]  # Placeholder for returning extra variables
        non_used_alphas_att = states[2]  # Placeholder for returning extra variables
        dp_mask = states[3]  # Dropout W (input)
        rec_dp_mask = states[4]  # Dropout U (recurrent)
        # Att model dropouts
        att_dp_mask = states[5]  # Dropout Wa
        pctx_ = states[6]  # Projected context (i.e. context * Ua + ba)
        context = states[7]  # Original context
        mask_context = states[8]  # Context mask
        if K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
            pctx_ = K.cast(mask_context[:, :, None], K.dtype(pctx_)) * pctx_
            context = K.cast(mask_context[:, :, None], K.dtype(context)) * context

        # GRU_1
        matrix_x_ = x
        if self.use_bias:
            matrix_x_ = K.bias_add(matrix_x_, self.bias1)
        matrix_inner_ = K.dot(h_tm1 * rec_dp_mask[0], self.recurrent1_kernel[:, :2 * self.units])
        x_z_ = matrix_x_[:, :self.units]
        x_r_ = matrix_x_[:, self.units: 2 * self.units]
        inner_z_ = matrix_inner_[:, :self.units]
        inner_r_ = matrix_inner_[:, self.units: 2 * self.units]
        z_ = self.recurrent_activation(x_z_ + inner_z_)
        r_ = self.recurrent_activation(x_r_ + inner_r_)
        x_h_ = matrix_x_[:, 2 * self.units:]
        inner_h_ = K.dot(r_ * h_tm1 * rec_dp_mask[0], self.recurrent1_kernel[:, 2 * self.units:])
        hh_ = self.activation(x_h_ + inner_h_)
        h_ = z_ * h_tm1 + (1 - z_) * hh_

        ctx_, alphas = compute_attention(h_, pctx_, context, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, mask_context,
                                         attention_mode=self.attention_mode)

        matrix_x = K.dot(ctx_ * dp_mask[0], self.kernel)
        if self.use_bias:
            matrix_x = K.bias_add(matrix_x, self.bias)
        matrix_inner = K.dot(h_ * rec_dp_mask[0], self.recurrent_kernel[:, :2 * self.units])

        x_z = matrix_x[:, :self.units]
        x_r = matrix_x[:, self.units: 2 * self.units]
        recurrent_z = matrix_inner[:, :self.units]
        recurrent_r = matrix_inner[:, self.units: 2 * self.units]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        x_h = matrix_x[:, 2 * self.units:]
        recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
                            self.recurrent_kernel[:, 2 * self.units:])
        hh = self.activation(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True

        return h, [h, ctx_, alphas]

    def get_constants(self, inputs, mask_context, training=None):
        constants = []
        # States[4] - Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training)
                       for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        # States[5] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        # States[6]  - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.attention_dropout < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, K.shape(self.context)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.attention_context_kernel)
        else:
            pctx = K.dot(self.context, self.attention_context_kernel)
        if self.use_bias:
            pctx = K.bias_add(pctx, self.bias_ba)
        # States[7] - pctx_
        constants.append(pctx)

        # States[8] - context
        constants.append(self.context)

        # States[9] - mask_context
        if mask_context is None:
            mask_context = K.not_equal(K.sum(self.context, axis=2), self.mask_value)
            mask_context = K.cast(mask_context, K.floatx())
        constants.append(mask_context)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        else:
            initial_state = self.init_state
        initial_states = [initial_state]

        initial_state = K.zeros_like(self.context)  # (samples, input_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)  # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)  # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states

    def get_config(self):
        config = {'return_extra_variables': self.return_extra_variables,
                  'return_states': self.return_states,
                  'units': self.units,
                  'att_units': self.att_units,
                  'mask_value': self.mask_value,
                  'use_bias': self.use_bias,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'conditional_initializer': initializers.serialize(self.conditional_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'attention_context_initializer': initializers.serialize(self.attention_context_initializer),
                  'attention_context_wa_initializer': initializers.serialize(self.attention_context_wa_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'conditional_regularizer': regularizers.serialize(self.conditional_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'attention_context_regularizer': regularizers.serialize(self.attention_context_regularizer),
                  'attention_context_wa_regularizer': regularizers.serialize(self.attention_context_wa_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'conditional_constraint': constraints.serialize(self.conditional_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'attention_context_constraint': constraints.serialize(self.attention_context_constraint),
                  'attention_context_wa_constraint': constraints.serialize(self.attention_context_wa_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'conditional_dropout': self.conditional_dropout,
                  'attention_dropout': self.attention_dropout,
                  'num_inputs': self.num_inputs,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttConditionalGRUCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMCond(Recurrent):
    """Conditional LSTM: The previously generated word is fed to the current timestep
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The input context  (shape: (batch_size, context_size))

    # Arguments
        units: Positive integer, dimensionality of the output space.
        return_states: Whether it should return the internal RNN states.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        forget_bias_init: Initializer for the forget bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        num_inputs: Number of inputs of the layer.
        static_ctx: If static_ctx, it should have 2 dimensions and it will
                    be fed to each timestep of the RNN. Otherwise, it should
                    have 3 dimensions and should have the same number of
                    timesteps than the input.
    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 return_states=False,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 forget_bias_init='one',
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 num_inputs=4,
                 static_ctx=False,
                 **kwargs):

        super(LSTMCond, self).__init__(**kwargs)

        self.return_states = return_states

        # Main parameters
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.forget_bias_initializer = initializers.get(forget_bias_init)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.num_inputs = num_inputs
        if static_ctx:
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        else:
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]

        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) == 2 or len(input_shape) == 4, 'You should pass two inputs to LSTMCond ' \
                                                               '(context and previous_embedded_words) and ' \
                                                               'two optional inputs (init_state and init_memory). ' \
                                                               'It currently has %d inputs' % len(input_shape)

        self.input_dim = input_shape[0][2]
        if self.input_spec[1].ndim == 3:
            self.context_dim = input_shape[1][2]
            self.static_ctx = False
            assert input_shape[1][1] == input_shape[0][1], 'When using a 3D ctx in LSTMCond, it has to have the same ' \
                                                           'number of timesteps (dimension 1) as the input. Currently,' \
                                                           'the number of input timesteps is: ' \
                                                           + str(input_shape[0][1]) + \
                                                           ', while the number of ctx timesteps is ' \
                                                           + str(input_shape[1][1]) + ' (complete shapes: ' \
                                                           + str(input_shape[0]) + ', ' + str(input_shape[1]) + ')'
        else:
            self.context_dim = input_shape[1][1]
            self.static_ctx = True

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None]  # [h, c]

        self.kernel = self.add_weight(shape=(self.context_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        else:
            self.bias = None

        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units))]

    def preprocess_input(self, inputs, training=None):
        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(4)]
            preprocessed_input = K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            preprocessed_input = K.dot(inputs, self.conditional_kernel)

        if self.static_ctx:
            return preprocessed_input

        # Not Static ctx
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs, ones,
                                        training=training) for _ in range(4)]
            preprocessed_context = K.dot(self.context * dp_mask[0][:, None, :], self.kernel)
        else:
            preprocessed_context = K.dot(self.context, self.kernel)
        return preprocessed_input + preprocessed_context

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_states:
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out = [main_out, states_dim, states_dim]
        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context = inputs[1]
        if self.num_inputs == 2:  # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3:  # input: [state_below, context, init_generic]
            self.init_state = inputs[2]
            self.init_memory = inputs[2]
        elif self.num_inputs == 4:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[2]
            self.init_memory = inputs[3]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0], states[1]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_sequences:
            ret = mask[0]
        else:
            ret = None
        if self.return_states:
            ret = [ret, None, None]
        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        dp_mask = states[2]  # Dropout W (input)
        rec_dp_mask = states[3]  # Dropout U (recurrent)
        z = x + K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
        if self.static_ctx:
            context = states[4]
            # mask_context = states[5]  # Context mask
            # if mask_context.ndim > 1:  # Mask the context (only if necessary)
            #    context = mask_context[:, :, None] * context
            z += K.dot(context * dp_mask[0], self.kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        o = self.recurrent_activation(z3)
        c = f * c_tm1 + i * self.activation(z2)
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]

    def get_constants(self, inputs, mask_context, training=None):
        constants = []

        # States[3] - Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[4] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[4] - context
        if self.static_ctx:
            constants.append(self.context)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        return initial_states

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'return_states': self.return_states,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'conditional_initializer': initializers.serialize(self.conditional_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'conditional_regularizer': regularizers.serialize(self.conditional_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'conditional_constraint': constraints.serialize(self.conditional_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'conditional_dropout': self.conditional_dropout,
                  'mask_value': self.mask_value,
                  'static_ctx': self.static_ctx,
                  'num_inputs': self.num_inputs
                  }
        base_config = super(LSTMCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttLSTM(Recurrent):
    """Long-Short Term Memory unit with Attention
    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        return_states: Whether it should return the internal RNN states.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        attention_mode: 'add', 'dot' or custom function.
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        forget_bias_init: Initializer for the forget gate
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
            Fraction of the units to drop for
            the linear transformation in the attended context.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        num_inputs: Number of inputs of the layer.


    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [units, input_dim] [input_dim]

        The names of 'Ua' and 'Wa' are exchanged w.r.t. the provided reference as well as 'v' being renamed
        to 'x' for matching Keras LSTM's nomenclature.

    # References
        -   Yao L, Torabi A, Cho K, Ballas N, Pal C, Larochelle H, Courville A.
            Describing videos by exploiting temporal structure.
            InProceedings of the IEEE International Conference on Computer Vision 2015 (pp. 4507-4515).
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 att_units=0,
                 return_extra_variables=False,
                 return_states=False,
                 activation='tanh',
                 attention_mode='add',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 forget_bias_init='one',
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 attention_recurrent_regularizer=None,
                 attention_context_regularizer=None,
                 attention_context_wa_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_context_constraint=None,
                 attention_context_wa_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 attention_dropout=0.,
                 num_inputs=3,
                 **kwargs):
        super(AttLSTM, self).__init__(**kwargs)
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.att_units = units if att_units == 0 else att_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.forget_bias_initializer = initializers.get(forget_bias_init)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) >= 2, 'You should pass two inputs to AttLSTM ' \
                                      '(previous_embedded_words and context) ' \
                                      'and two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[0][1]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None]  # [h, c, x_att]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.attention_recurrent_kernel = self.add_weight(
            shape=(self.units, self.att_units),
            name='attention_recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(
            shape=(self.input_dim, self.att_units),
            name='attention_context_kernel',
            initializer=self.attention_context_initializer,
            regularizer=self.attention_context_regularizer,
            constraint=self.attention_context_constraint)

        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(
                shape=(self.att_units,),
                name='attention_context_wa',
                initializer=self.attention_context_wa_initializer,
                regularizer=self.attention_context_wa_regularizer,
                constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.bias_ba = self.add_weight(shape=(self.att_units,),
                                       name='bias_ba',
                                       initializer=self.bias_ba_initializer,
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           name='bias_ca',
                                           initializer=self.bias_ca_initializer,
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None

        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):
        return inputs

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim, states_dim]

        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        if self.num_inputs == 1:  # input: [context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 2:  # input: [context, init_generic]
            self.init_state = inputs[1]
            self.init_memory = inputs[1]
        elif self.num_inputs == 3:  # input: [context, init_state, init_memory]
            self.init_state = inputs[1]
            self.init_memory = inputs[2]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[2, 3])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[2], states[3]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0], states[1]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]

        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        non_used_x_att = states[2]  # Placeholder for returning extra variables
        non_used_alphas_att = states[3]  # Placeholder for returning extra variables
        dp_mask = states[4]  # Dropout W (input)
        rec_dp_mask = states[5]  # Dropout U (recurrent)
        # Att model dropouts
        att_dp_mask = states[6]  # Dropout Wa
        pctx_ = states[7]  # Projected context (i.e. context * Ua + ba)

        # Attention model (see Formulation in class header)
        ctx_, alphas = compute_attention(h_tm1, pctx_, pctx_, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, None,
                                         attention_mode=self.attention_mode)
        # LSTM
        z = x + \
            K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel) + \
            K.dot(ctx_ * dp_mask[0], self.kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        o = self.recurrent_activation(z3)
        c = f * c_tm1 + i * self.activation(z2)
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c, ctx_, alphas]

    def get_constants(self, inputs, training=None):
        constants = []
        # States[4] - Dropout W (input dropout)
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[5] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[6]  - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.attention_dropout < 1:
            input_dim = K.shape(inputs)[2]
            ones = K.ones_like(K.reshape(inputs[:, :, 0], (-1, K.shape(inputs)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx = K.dot(inputs * B_Ua[0], self.attention_context_kernel)
        else:
            pctx = K.dot(inputs, self.attention_context_kernel)
        if self.use_bias:
            pctx = K.bias_add(pctx, self.bias_ba)
        # States[7] - pctx_
        constants.append(pctx)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        initial_state = K.zeros_like(inputs)  # (samples, input_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)  # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)  # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states

    def get_config(self):
        config = {'units': self.units,
                  "att_units": self.att_units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'return_extra_variables': self.return_extra_variables,
                  'return_states': self.return_states,
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'attention_dropout': self.attention_dropout,
                  'mask_value': self.mask_value,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttLSTMCond(Recurrent):
    """Long-Short Term Memory unit with Attention + the previously generated word fed to the current timestep.

    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        return_states: Whether it should return the internal RNN states.
        attention_mode: 'add', 'dot' or custom function.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        num_inputs: Number of inputs of the layer.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 att_units=0,
                 return_extra_variables=False,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 attention_recurrent_regularizer=None,
                 attention_context_regularizer=None,
                 attention_context_wa_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_context_constraint=None,
                 attention_context_wa_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 num_inputs=4,
                 **kwargs):
        super(AttLSTMCond, self).__init__(**kwargs)
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.att_units = units if att_units == 0 else att_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attention_mode = attention_mode.lower()
        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.unit_forget_bias = unit_forget_bias

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) >= 2, 'You should pass two inputs to AttLSTMCond ' \
                                      '(previous_embedded_words and context) ' \
                                      'and two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[1][1]
        self.context_dim = input_shape[1][2]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None]  # [h, c, x_att]

        self.kernel = self.add_weight(shape=(self.context_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(
            shape=(self.units, self.att_units),
            name='attention_recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(
            shape=(self.context_dim, self.att_units),
            name='attention_context_kernel',
            initializer=self.attention_context_initializer,
            regularizer=self.attention_context_regularizer,
            constraint=self.attention_context_constraint)

        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(
                shape=(self.att_units,),
                name='attention_context_wa',
                initializer=self.attention_context_wa_initializer,
                regularizer=self.attention_context_wa_regularizer,
                constraint=self.attention_context_wa_constraint)

        else:
            self.attention_context_wa = None

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.bias_ba = self.add_weight(shape=(self.att_units,),
                                       name='bias_ba',
                                       initializer=self.bias_ba_initializer,
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           name='bias_ca',
                                           initializer=self.bias_ca_initializer,
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None
        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):

        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(4)]
            return K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            return K.dot(inputs, self.conditional_kernel)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim, states_dim]

        return main_out

    def _ln(self, x, slc):
        # sample-wise normalization
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon_layer_normalization)
        x_normed = (x - m) / (std + self.epsilon_layer_normalization)
        x_normed = eval('self.gamma_' + slc) * x_normed + eval('self.beta_' + slc)
        return x_normed

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context = inputs[1]
        if self.num_inputs == 2:  # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3:  # input: [state_below, context, init_generic]
            self.init_state = inputs[2]
            self.init_memory = inputs[2]
        elif self.num_inputs == 4:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[2]
            self.init_memory = inputs[3]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[2, 3])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[2], states[3]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0], states[1]]
        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]

        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        non_used_x_att = states[2]  # Placeholder for returning extra variables
        non_used_alphas_att = states[3]  # Placeholder for returning extra variables
        dp_mask = states[4]  # Dropout W (input)
        rec_dp_mask = states[5]  # Dropout U (recurrent)
        # Att model dropouts
        att_dp_mask = states[6]  # Dropout Wa
        pctx_ = states[7]  # Projected context (i.e. context * Ua + ba)
        context = states[8]  # Original context
        mask_context = states[9]  # Context mask
        if K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
            pctx_ = K.cast(mask_context[:, :, None], K.dtype(pctx_)) * pctx_
            context = K.cast(mask_context[:, :, None], K.dtype(context)) * context

        ctx_, alphas = compute_attention(h_tm1, pctx_, context, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, mask_context,
                                         attention_mode=self.attention_mode)
        # LSTM
        z = x + \
            K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel) + \
            K.dot(ctx_ * dp_mask[0], self.kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        o = self.recurrent_activation(z3)
        c = f * c_tm1 + i * self.activation(z2)
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c, ctx_, alphas]

    def get_constants(self, inputs, mask_context, training=None):
        constants = []
        # States[4] - Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[5] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[6]  - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.attention_dropout < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, K.shape(self.context)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.attention_context_kernel)
        else:
            pctx = K.dot(self.context, self.attention_context_kernel)
        if self.use_bias:
            pctx = K.bias_add(pctx, self.bias_ba)
        # States[7] - pctx_
        constants.append(pctx)

        # States[8] - context
        constants.append(self.context)

        # States[9] - mask_context
        if mask_context is None:
            mask_context = K.not_equal(K.sum(self.context, axis=2), self.mask_value)
            mask_context = K.cast(mask_context, K.floatx())
        constants.append(mask_context)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        initial_state = K.zeros_like(self.context)  # (samples, input_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)  # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)  # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states

    def get_config(self):
        config = {'return_extra_variables': self.return_extra_variables,
                  'return_states': self.return_states,
                  'units': self.units,
                  "att_units": self.att_units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'mask_value': self.mask_value,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'conditional_initializer': initializers.serialize(self.conditional_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'attention_context_initializer': initializers.serialize(self.attention_context_initializer),
                  'attention_context_wa_initializer': initializers.serialize(self.attention_context_wa_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'conditional_regularizer': regularizers.serialize(self.conditional_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'attention_context_regularizer': regularizers.serialize(self.attention_context_regularizer),
                  'attention_context_wa_regularizer': regularizers.serialize(self.attention_context_wa_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'conditional_constraint': constraints.serialize(self.conditional_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'attention_context_constraint': constraints.serialize(self.attention_context_constraint),
                  'attention_context_wa_constraint': constraints.serialize(self.attention_context_wa_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'conditional_dropout': self.conditional_dropout,
                  'attention_dropout': self.attention_dropout,
                  'num_inputs': self.num_inputs,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttLSTMCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttConditionalLSTMCond(Recurrent):
    """Conditional Long-Short Term Memory unit with Attention + the previously generated word fed to the current timestep.

    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        att_mode: Attention mode. 'add' or 'dot' implemented.
        return_states: Whether it should return the internal RNN states.
        attention_mode: 'add', 'dot' or custom function.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        num_inputs: Number of inputs of the layer.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Nematus: a Toolkit for Neural Machine Translation](http://arxiv.org/abs/1703.04357)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 att_units=0,
                 return_extra_variables=False,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 attention_recurrent_regularizer=None,
                 attention_context_regularizer=None,
                 attention_context_wa_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_context_constraint=None,
                 attention_context_wa_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 num_inputs=4,
                 **kwargs):
        super(AttConditionalLSTMCond, self).__init__(**kwargs)

        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.att_units = units if att_units == 0 else att_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.recurrent1_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias1_initializer = initializers.get(bias_initializer)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.unit_forget_bias = unit_forget_bias

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.recurrent1_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias1_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.recurrent1_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias1_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.

        # Inputs
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):
        assert len(input_shape) >= 2, 'You should pass two inputs to AttConditionalLSTMCond ' \
                                      '(previous_embedded_words and context) ' \
                                      'and two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[1][1]
        self.context_dim = input_shape[1][2]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None]  # [h, c, x_att]

        self.kernel = self.add_weight(shape=(self.context_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent1_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent1_kernel',
            initializer=self.recurrent1_initializer,
            regularizer=self.recurrent1_regularizer,
            constraint=self.recurrent1_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(
            shape=(self.units, self.att_units),
            name='attention_recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(
            shape=(self.context_dim, self.att_units),
            name='attention_context_kernel',
            initializer=self.attention_context_initializer,
            regularizer=self.attention_context_regularizer,
            constraint=self.attention_context_constraint)

        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(
                shape=(self.att_units,),
                name='attention_context_wa',
                initializer=self.attention_context_wa_initializer,
                regularizer=self.attention_context_wa_regularizer,
                constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if self.unit_forget_bias:
                def bias_initializer1(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias1_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias1_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer1 = self.bias1_initializer
            self.bias1 = self.add_weight(shape=(self.units * 4,),
                                         name='bias1',
                                         initializer=bias_initializer1,
                                         regularizer=self.bias1_regularizer,
                                         constraint=self.bias1_constraint)

            self.bias_ba = self.add_weight(shape=(self.att_units,),
                                           name='bias_ba',
                                           initializer=self.bias_ba_initializer,
                                           regularizer=self.bias_ba_regularizer,
                                           constraint=self.bias_ba_constraint)

            if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
                bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
                self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                               name='bias_ca',
                                               initializer=self.bias_ca_initializer,
                                               regularizer=self.bias_ca_regularizer,
                                               constraint=self.bias_ca_constraint)
            else:
                self.bias_ca = None

        else:
            self.bias = None
            self.bias1 = None
            self.bias_ba = None
            self.bias_ca = None

        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):

        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(4)]
            return K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            return K.dot(inputs, self.conditional_kernel)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim, states_dim]

        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context = inputs[1]
        if self.num_inputs == 2:  # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3:  # input: [state_below, context, init_generic]
            self.init_state = inputs[2]
            self.init_memory = inputs[2]
        elif self.num_inputs == 4:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[2]
            self.init_memory = inputs[3]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[2, 3])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)
        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            ret._uses_learning_phase = True

        if self.return_extra_variables:
            ret = [ret, states[2], states[3]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, (list, tuple)):
                ret = [ret]
            else:
                states = list(states)
            ret += [states[0], states[1]]

        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]

        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        non_used_x_att = states[2]  # Placeholder for returning extra variables
        non_used_alphas_att = states[3]  # Placeholder for returning extra variables
        ctx_dp_mask = states[4]  # Dropout W
        rec_dp_mask = states[5]  # Dropout U
        # Att model dropouts
        att_dp_mask = states[6]  # Dropout Wa
        pctx_ = states[7]  # Projected context (i.e. context * Ua + ba)
        context = states[8]  # Original context
        mask_context = states[9]  # Context mask
        if K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
            pctx_ = K.cast(mask_context[:, :, None], K.dtype(pctx_)) * pctx_
            context = K.cast(mask_context[:, :, None], K.dtype(context)) * context

        # LSTM_1
        z_ = x + K.dot(h_tm1 * rec_dp_mask[0], self.recurrent1_kernel)
        if self.use_bias:
            z_ = K.bias_add(z_, self.bias1)
        z_0 = z_[:, :self.units]
        z_1 = z_[:, self.units: 2 * self.units]
        z_2 = z_[:, 2 * self.units: 3 * self.units]
        z_3 = z_[:, 3 * self.units:]
        i_ = self.recurrent_activation(z_0)
        f_ = self.recurrent_activation(z_1)
        o_ = self.recurrent_activation(z_3)
        c_ = f_ * c_tm1 + i_ * self.activation(z_2)
        h_ = o_ * self.activation(c_)

        ctx_, alphas = compute_attention(h_, pctx_, context, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, mask_context,
                                         attention_mode=self.attention_mode)

        # LSTM
        z = K.dot(h_ * rec_dp_mask[0], self.recurrent_kernel) + \
            K.dot(ctx_ * ctx_dp_mask[0], self.kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        o = self.recurrent_activation(z3)
        c = f * c_ + i * self.activation(z2)
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c, ctx_, alphas]

    def get_constants(self, inputs, mask_context, training=None):
        constants = []
        # States[4] - Dropout W (input dropout)
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[5] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[6]  - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.attention_dropout < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, K.shape(self.context)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.attention_context_kernel)
        else:
            pctx = K.dot(self.context, self.attention_context_kernel)
        if self.use_bias:
            pctx = K.bias_add(pctx, self.bias_ba)
        # States[7] - pctx_
        constants.append(pctx)

        # States[8] - context
        constants.append(self.context)

        # States[9] - mask_context
        if mask_context is None:
            mask_context = K.not_equal(K.sum(self.context, axis=2), self.mask_value)
            mask_context = K.cast(mask_context, K.floatx())
        constants.append(mask_context)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        initial_state = K.zeros_like(self.context)  # (samples, input_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)  # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)  # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states

    def get_config(self):
        config = {'return_extra_variables': self.return_extra_variables,
                  'return_states': self.return_states,
                  'units': self.units,
                  'att_units': self.att_units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'mask_value': self.mask_value,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'conditional_initializer': initializers.serialize(self.conditional_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'attention_context_initializer': initializers.serialize(self.attention_context_initializer),
                  'attention_context_wa_initializer': initializers.serialize(self.attention_context_wa_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'conditional_regularizer': regularizers.serialize(self.conditional_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'attention_context_regularizer': regularizers.serialize(self.attention_context_regularizer),
                  'attention_context_wa_regularizer': regularizers.serialize(self.attention_context_wa_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'conditional_constraint': constraints.serialize(self.conditional_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'attention_context_constraint': constraints.serialize(self.attention_context_constraint),
                  'attention_context_wa_constraint': constraints.serialize(self.attention_context_wa_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'conditional_dropout': self.conditional_dropout,
                  'attention_dropout': self.attention_dropout,
                  'num_inputs': self.num_inputs,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttConditionalLSTMCond, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttLSTMCond2Inputs(Recurrent):
    """Long-Short Term Memory unit with the previously generated word fed to the current timestep
    and two input contexts (with two attention mechanisms).

    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units1:  Positive integer, dimensionality of the first attention space.
        att_units2:  Positive integer, dimensionality of the second attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        attend_on_both: Boolean, wether attend on both inputs or not.
        return_states: Whether it should return the internal RNN states.
        attention_mode: 'add', 'dot' or custom function.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        kernel_initializer2: Initializer for the `kernel2` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer2:  Initializer for the `attention_recurrent_kernel2`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer2:  Initializer for the `attention_context_kernel2`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer2:  Initializer for the `attention_wa_kernel2`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_initializer2: Initializer for the bias vector 2
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ba_initializer2: Initializer for the bias_ba2 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer2: Initializer for the bias_ca2 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_regularizer2: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_regularizer2: Regularizer function applied to the bias2 vector
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer2:  Regularizer function applied to
            the `attention_recurrent__kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer2:  Regularizer function applied to
            the `attention_context_kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer2:  Regularizer function applied to
            the `attention_context_wa_kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer2:  Regularizer function applied to the bias_ba2 vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer2:  Regularizer function applied to the bias_ca2 vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        kernel_constraint2: Constraint function applied to
            the `kernel2` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint2: Constraint function applied to
            the `attention_recurrent_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint2: Constraint function applied to
            the `attention_context_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint2: Constraint function applied to
            the `attention_context_wa_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_constraint2: Constraint function applied to the bias2 vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ba_constraint2: Constraint function applied to
            the `bias_ba2` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint2: Constraint function applied to
            the `bias_ca2` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        dropout2: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context2.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        attention_dropout2: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism2.
        num_inputs: Number of inputs of the layer.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Egocentric Video Description based on Temporally-Linked Sequences](https://arxiv.org/abs/1704.02163)
    """

    def __init__(self, units,
                 att_units1=0,
                 att_units2=0,
                 return_extra_variables=False,
                 attend_on_both=False,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_initializer='glorot_uniform',
                 kernel_initializer2='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_recurrent_initializer2='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_initializer2='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 attention_context_wa_initializer2='glorot_uniform',
                 bias_initializer='zeros',
                 bias_initializer2='zeros',
                 bias_ba_initializer='zeros',
                 bias_ba_initializer2='zeros',
                 bias_ca_initializer='zero',
                 bias_ca_initializer2='zero',
                 kernel_regularizer=None,
                 kernel_regularizer2=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 bias_regularizer=None,
                 bias_regularizer2=None,
                 attention_recurrent_regularizer=None,
                 attention_recurrent_regularizer2=None,
                 attention_context_regularizer=None,
                 attention_context_regularizer2=None,
                 attention_context_wa_regularizer=None,
                 attention_context_wa_regularizer2=None,
                 bias_ba_regularizer=None,
                 bias_ba_regularizer2=None,
                 bias_ca_regularizer=None,
                 bias_ca_regularizer2=None,
                 kernel_constraint=None,
                 kernel_constraint2=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_recurrent_constraint2=None,
                 attention_context_constraint=None,
                 attention_context_constraint2=None,
                 attention_context_wa_constraint=None,
                 attention_context_wa_constraint2=None,
                 bias_constraint=None,
                 bias_constraint2=None,
                 bias_ba_constraint=None,
                 bias_ba_constraint2=None,
                 bias_ca_constraint=None,
                 bias_ca_constraint2=None,
                 dropout=0.,
                 dropout2=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 attention_dropout2=0.,
                 num_inputs=5,
                 **kwargs):

        super(AttLSTMCond2Inputs, self).__init__(**kwargs)

        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.num_inputs = num_inputs
        self.att_units1 = units if att_units1 == 0 else att_units1
        self.att_units2 = units if att_units2 == 0 else att_units2
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attend_on_both = attend_on_both
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer2 = initializers.get(kernel_initializer2)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_recurrent_initializer2 = initializers.get(attention_recurrent_initializer2)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_initializer2 = initializers.get(attention_context_initializer2)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.attention_context_wa_initializer2 = initializers.get(attention_context_wa_initializer2)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_initializer2 = initializers.get(bias_initializer2)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ba_initializer2 = initializers.get(bias_ba_initializer2)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.bias_ca_initializer2 = initializers.get(bias_ca_initializer2)
        self.unit_forget_bias = unit_forget_bias

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_regularizer2 = regularizers.get(kernel_regularizer2)
        self.bias_regularizer2 = regularizers.get(bias_regularizer2)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        # attention model learnable params
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        if self.attend_on_both:
            # attention model 2 learnable params
            self.attention_context_wa_regularizer2 = regularizers.get(attention_context_wa_regularizer2)
            self.attention_context_regularizer2 = regularizers.get(attention_context_regularizer2)
            self.attention_recurrent_regularizer2 = regularizers.get(attention_recurrent_regularizer2)
            self.bias_ba_regularizer2 = regularizers.get(bias_ba_regularizer2)
            self.bias_ca_regularizer2 = regularizers.get(bias_ca_regularizer2)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_constraint2 = constraints.get(kernel_constraint2)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_recurrent_constraint2 = constraints.get(attention_recurrent_constraint2)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_constraint2 = constraints.get(attention_context_constraint2)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.attention_context_wa_constraint2 = constraints.get(attention_context_wa_constraint2)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_constraint2 = constraints.get(bias_constraint2)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ba_constraint2 = constraints.get(bias_ba_constraint2)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)
        self.bias_ca_constraint2 = constraints.get(bias_ca_constraint2)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.dropout2 = min(1., max(0., dropout2)) if dropout2 is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        if self.attend_on_both:
            self.attention_dropout2 = min(1., max(0., attention_dropout2)) if attention_dropout2 is not None else 0.

        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):
        assert len(input_shape) >= 3 or 'You should pass three inputs to AttLSTMCond2Inputs ' \
                                        '(previous_embedded_words, context1 and context2) and ' \
                                        'two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None, None]  # [h, c, x_att, x_att2]

        if self.attend_on_both:
            assert K.ndim(self.input_spec[1]) == 3 and K.ndim(self.input_spec[2]), 'When using two attention models,' \
                                                                                   'you should pass two 3D tensors' \
                                                                                   'to AttLSTMCond2Inputs'
        else:
            assert K.ndim(self.input_spec[1]) == 3, 'When using an attention model, you should pass one 3D tensors' \
                                                    'to AttLSTMCond2Inputs'

        if K.ndim(self.input_spec[1]) == 3:
            self.context1_steps = input_shape[1][1]
            self.context1_dim = input_shape[1][2]

        if K.ndim(self.input_spec[2]) == 3:
            self.context2_steps = input_shape[2][1]
            self.context2_dim = input_shape[2][2]
        else:
            self.context2_dim = input_shape[2][1]

        self.kernel = self.add_weight(shape=(self.context1_dim, self.units * 4),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel2 = self.add_weight(shape=(self.context2_dim, self.units * 4),
                                       initializer=self.kernel_initializer2,
                                       name='kernel2',
                                       regularizer=self.kernel_regularizer2,
                                       constraint=self.kernel_constraint2)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(shape=(self.units, self.att_units1),
                                                          initializer=self.attention_recurrent_initializer,
                                                          name='attention_recurrent_kernel',
                                                          regularizer=self.attention_recurrent_regularizer,
                                                          constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(shape=(self.context1_dim, self.att_units1),
                                                        initializer=self.attention_context_initializer,
                                                        name='attention_context_kernel',
                                                        regularizer=self.attention_context_regularizer,
                                                        constraint=self.attention_context_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(shape=(self.att_units1,),
                                                        initializer=self.attention_context_wa_initializer,
                                                        name='attention_context_wa',
                                                        regularizer=self.attention_context_wa_regularizer,
                                                        constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None
        self.bias_ba = self.add_weight(shape=(self.att_units1,),
                                       initializer=self.bias_ba_initializer,
                                       name='bias_ba',
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':

            bias_ca_shape = self.context1_steps if self.context1_steps is None else (self.context1_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           initializer=self.bias_ca_initializer,
                                           name='bias_ca',
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            self.bias_ba = None
            self.bias_ca = None

        if self.attend_on_both:
            # Initialize Att model params (following the same format for any option of self.consume_less)
            self.attention_recurrent_kernel2 = self.add_weight(shape=(self.units, self.att_units2),
                                                               initializer=self.attention_recurrent_initializer2,
                                                               name='attention_recurrent_kernel2',
                                                               regularizer=self.attention_recurrent_regularizer2,
                                                               constraint=self.attention_recurrent_constraint2)

            self.attention_context_kernel2 = self.add_weight(shape=(self.context2_dim, self.att_units2),
                                                             initializer=self.attention_context_initializer2,
                                                             name='attention_context_kernel2',
                                                             regularizer=self.attention_context_regularizer2,
                                                             constraint=self.attention_context_constraint2)
            if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
                self.attention_context_wa2 = self.add_weight(shape=(self.att_units2,),
                                                             initializer=self.attention_context_wa_initializer2,
                                                             name='attention_context_wa2',
                                                             regularizer=self.attention_context_wa_regularizer2,
                                                             constraint=self.attention_context_wa_constraint2)
            else:
                self.attention_context_wa2 = None

            self.bias_ba2 = self.add_weight(shape=(self.att_units2,),
                                            initializer=self.bias_ba_initializer2,
                                            name='bias_ba2',
                                            regularizer=self.bias_ba_regularizer2,
                                            constraint=self.bias_ba_constraint2)
            if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
                bias_ca_shape = self.context2_steps if self.context2_steps is None else (self.context2_steps,)
                self.bias_ca2 = self.add_weight(shape=bias_ca_shape,
                                                initializer=self.bias_ca_initializer2,
                                                name='bias_ca2',
                                                regularizer=self.bias_ca_regularizer2,
                                                constraint=self.bias_ca_constraint2)
            else:
                self.bias_ca2 = None
            if self.use_bias:
                if self.unit_forget_bias:
                    def bias_initializer2(shape, *args, **kwargs):
                        return K.concatenate([
                            self.bias_initializer2((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer2((self.units * 2,), *args, **kwargs),
                        ])
                else:
                    bias_initializer2 = self.bias_initializer2
                self.bias2 = self.add_weight(shape=(self.units * 4,),
                                             name='bias2',
                                             initializer=bias_initializer2,
                                             regularizer=self.bias_regularizer2,
                                             constraint=self.bias_constraint2)
            else:
                self.bias2 = None
                self.bias_ba2 = None
                self.bias_ca2 = None

        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = K.shape(self.input_spec[0][0])
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):
        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(4)]
            return K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            return K.dot(inputs, self.conditional_kernel)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context1_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            dim_x_att2 = (input_shape[0][0], input_shape[0][1], self.context2_dim)
            dim_alpha_att2 = (input_shape[0][0], input_shape[0][1], input_shape[2][1])
            main_out = [main_out, dim_x_att, dim_alpha_att, dim_x_att2, dim_alpha_att2]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim, states_dim]

        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context1 = inputs[1]
        self.context2 = inputs[2]
        if self.num_inputs == 3:  # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 4:  # input: [state_below, context, init_generic]
            self.init_state = inputs[3]
            self.init_memory = inputs[3]
        elif self.num_inputs == 5:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[3]
            self.init_memory = inputs[4]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], mask[2], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[2, 3, 4, 5])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output
        if self.return_extra_variables:
            ret = [ret, states[2], states[3], states[4], states[5]]
        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0], states[1]]

        return ret

    def compute_mask(self, input, mask):

        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]
        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        pos_states = 10

        non_used_x_att = states[2]  # Placeholder for returning extra variables
        non_used_alphas_att = states[3]  # Placeholder for returning extra variables
        non_used_x_att2 = states[4]  # Placeholder for returning extra variables
        non_used_alphas_att2 = states[5]  # Placeholder for returning extra variables

        rec_dp_mask = states[6]  # Dropout U
        dp_mask2 = states[7]  # Dropout T
        dp_mask = states[8]  # Dropout W

        # Att model dropouts
        # att_dp_mask_wa = states[9]  # Dropout wa
        att_dp_mask = states[9]  # Dropout Wa
        # Att model 2 dropouts
        if self.attend_on_both:
            # att_dp_mask_wa2 = states[pos_states]  # Dropout wa
            att_dp_mask2 = states[pos_states]  # Dropout Wa

            context1 = states[pos_states + 1]  # Context
            mask_context1 = states[pos_states + 2]  # Context mask
            pctx_1 = states[pos_states + 3]  # Projected context (i.e. context * Ua + ba)

            context2 = states[pos_states + 4]  # Context 2
            mask_context2 = states[pos_states + 5]  # Context 2 mask
            pctx_2 = states[pos_states + 6]  # Projected context 2 (i.e. context * Ua2 + ba2)
        else:
            context1 = states[pos_states]  # Context
            mask_context1 = states[pos_states + 1]  # Context mask
            pctx_1 = states[pos_states + 2]  # Projected context (i.e. context * Ua + ba)

            context2 = states[pos_states + 3]  # Context 2
            mask_context2 = states[pos_states + 4]  # Context 2 mask

        if K.ndim(mask_context1) > 1:  # Mask the context (only if necessary)
            pctx_1 = mask_context1[:, :, None] * pctx_1
            context1 = mask_context1[:, :, None] * context1

        ctx_1, alphas1 = compute_attention(h_tm1, pctx_1, context1, att_dp_mask, self.attention_recurrent_kernel,
                                           self.attention_context_wa, self.bias_ca, mask_context1,
                                           attention_mode=self.attention_mode)

        if self.attend_on_both:
            if K.ndim(mask_context2) > 1:  # Mask the context2 (only if necessary)
                pctx_2 = mask_context2[:, :, None] * pctx_2
                context2 = mask_context2[:, :, None] * context2

            # Attention model 2 (see Formulation in class header)
            ctx_2, alphas2 = compute_attention(h_tm1, pctx_1, context2, att_dp_mask2, self.attention_recurrent_kernel2,
                                               self.attention_context_wa2, self.bias_ca2, mask_context2,
                                               attention_mode=self.attention_mode)
        else:
            ctx_2 = context2
            alphas2 = mask_context2

        z = x + \
            K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel) + \
            K.dot(ctx_2 * dp_mask2[0], self.kernel2) + \
            K.dot(ctx_1 * dp_mask[0], self.kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
            if self.attend_on_both:
                z = K.bias_add(z, self.bias2)
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        h = o * self.activation(c)
        return h, [h, c, ctx_1, alphas1, ctx_2, alphas2]

    def get_constants(self, inputs, mask_context1, mask_context2, training=None):
        constants = []
        # States[6] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[7]- Dropout_T
        if 0 < self.dropout2 < 1:
            ones = K.ones_like(K.squeeze(self.context2[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout2)

            dp_mask2 = [K.in_train_phase(dropped_inputs,
                                         ones,
                                         training=training) for _ in range(4)]
            constants.append(dp_mask2)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[8]- Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context1[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # AttModel
        # States[9] - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if self.attend_on_both:
            # AttModel2
            # States[10]
            if 0 < self.attention_dropout2 < 1:
                input_dim = self.units
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * input_dim, 1)

                def dropped_inputs():
                    return K.dropout(ones, self.recurrent_dropout)

                att_dp_mask2 = [K.in_train_phase(dropped_inputs,
                                                 ones,
                                                 training=training)]
                constants.append(att_dp_mask2)
            else:
                constants.append([K.cast_to_floatx(1.)])

        # States[11] - Context1
        constants.append(self.context1)
        # States[12] - MaskContext1
        if mask_context1 is None:
            mask_context1 = K.not_equal(K.sum(self.context1, axis=2), self.mask_value)
            mask_context1 = K.cast(mask_context1, K.floatx())
        constants.append(mask_context1)

        # States[13] - pctx_1
        if 0 < self.attention_dropout < 1:
            input_dim = self.context1_dim
            ones = K.ones_like(K.reshape(self.context1[:, :, 0], (-1, K.shape(self.context1)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx_1 = K.dot(self.context1 * B_Ua[0], self.attention_context_kernel)
        else:
            pctx_1 = K.dot(self.context1, self.attention_context_kernel)
        if self.use_bias:
            pctx_1 = K.bias_add(pctx_1, self.bias_ba)
        constants.append(pctx_1)

        if self.attend_on_both:

            # States[14] - Context2
            constants.append(self.context2)
            # States[15] - MaskContext2
            if self.attend_on_both:
                if mask_context2 is None:
                    mask_context2 = K.not_equal(K.sum(self.context2, axis=2), self.mask_value)
                    mask_context2 = K.cast(mask_context2, K.floatx())
            else:
                mask_context2 = K.ones_like(self.context2[:, 0])
            constants.append(mask_context2)
            # States[16] - pctx_2
            if 0 < self.attention_dropout2 < 1:
                input_dim = self.context2_dim
                ones = K.ones_like(K.reshape(self.context2[:, :, 0], (-1, K.shape(self.context2)[1], 1)))
                ones = K.concatenate([ones] * input_dim, axis=2)
                B_Ua2 = [K.in_train_phase(K.dropout(ones, self.attention_dropout2), ones)]
                pctx_2 = K.dot(self.context2 * B_Ua2[0], self.attention_context_kernel2)
            else:
                pctx_2 = K.dot(self.context2, self.attention_context_kernel2)
            if self.use_bias:
                pctx_2 = K.bias_add(pctx_2, self.bias_ba2)
            constants.append(pctx_2)

        return constants

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            # build an all-zero tensor of shape (samples, units)
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        # extra states for context1 and context2
        initial_state1 = K.zeros_like(self.context1)  # (samples, input_timesteps, ctx1_dim)
        initial_state_alphas1 = K.sum(initial_state1, axis=2)  # (samples, input_timesteps)
        initial_state1 = K.sum(initial_state1, axis=1)  # (samples, ctx1_dim)
        extra_states = [initial_state1, initial_state_alphas1]
        initial_state2 = K.zeros_like(self.context2)  # (samples, input_timesteps, ctx2_dim)
        if self.attend_on_both:  # Reduce on temporal dimension
            initial_state_alphas2 = K.sum(initial_state2, axis=2)  # (samples, input_timesteps)
            initial_state2 = K.sum(initial_state2, axis=1)  # (samples, ctx2_dim)
        else:  # Already reduced
            initial_state_alphas2 = initial_state2  # (samples, ctx2_dim)

        extra_states.append(initial_state2)
        extra_states.append(initial_state_alphas2)

        return initial_states + extra_states

    def get_config(self):
        config = {"units": self.units,
                  "att_units1": self.att_units1,
                  "att_units2": self.att_units2,
                  "return_extra_variables": self.return_extra_variables,
                  "return_states": self.return_states,
                  "use_bias": self.use_bias,
                  "mask_value": self.mask_value,
                  "attend_on_both": self.attend_on_both,
                  'unit_forget_bias': self.unit_forget_bias,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                  "kernel_regularizer2": regularizers.serialize(self.kernel_regularizer2),
                  "conditional_regularizer": regularizers.serialize(self.conditional_regularizer),
                  "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
                  "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                  'attention_context_wa_regularizer': regularizers.serialize(self.attention_context_wa_regularizer),
                  'attention_context_regularizer': regularizers.serialize(self.attention_context_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  "bias_regularizer2": regularizers.serialize(self.bias_regularizer2),
                  'attention_context_wa_regularizer2': regularizers.serialize(self.attention_context_wa_regularizer2),
                  'attention_context_regularizer2': regularizers.serialize(self.attention_context_regularizer2),
                  'attention_recurrent_regularizer2': regularizers.serialize(self.attention_recurrent_regularizer2),
                  'bias_ba_regularizer2': regularizers.serialize(self.bias_ba_regularizer2),
                  'bias_ca_regularizer2': regularizers.serialize(self.bias_ca_regularizer2),
                  "kernel_initializer": initializers.serialize(self.kernel_initializer),
                  "kernel_initializer2": initializers.serialize(self.kernel_initializer2),
                  "conditional_initializer": initializers.serialize(self.conditional_initializer),
                  "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
                  "bias_initializer": initializers.serialize(self.bias_initializer),
                  'attention_context_wa_initializer': initializers.serialize(self.attention_context_wa_initializer),
                  'attention_context_initializer': initializers.serialize(self.attention_context_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  "bias_initializer2": initializers.serialize(self.bias_initializer2),
                  'attention_context_wa_initializer2': initializers.serialize(self.attention_context_wa_initializer2),
                  'attention_context_initializer2': initializers.serialize(self.attention_context_initializer2),
                  'attention_recurrent_initializer2': initializers.serialize(self.attention_recurrent_initializer2),
                  'bias_ba_initializer2': initializers.serialize(self.bias_ba_initializer2),
                  'bias_ca_initializer2': initializers.serialize(self.bias_ca_initializer2),
                  "kernel_constraint": constraints.serialize(self.kernel_constraint),
                  "kernel_constraint2": constraints.serialize(self.kernel_constraint2),
                  "conditional_constraint": constraints.serialize(self.conditional_constraint),
                  "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
                  "bias_constraint": constraints.serialize(self.bias_constraint),
                  'attention_context_wa_constraint': constraints.serialize(self.attention_context_wa_constraint),
                  'attention_context_constraint': constraints.serialize(self.attention_context_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  "bias_constraint2": constraints.serialize(self.bias_constraint2),
                  'attention_context_wa_constraint2': constraints.serialize(self.attention_context_wa_constraint2),
                  'attention_context_constraint2': constraints.serialize(self.attention_context_constraint2),
                  'attention_recurrent_constraint2': constraints.serialize(self.attention_recurrent_constraint2),
                  'bias_ba_constraint2': constraints.serialize(self.bias_ba_constraint2),
                  'bias_ca_constraint2': constraints.serialize(self.bias_ca_constraint2),
                  "dropout": self.dropout,
                  "dropout2": self.dropout2,
                  "recurrent_dropout": self.recurrent_dropout,
                  "conditional_dropout": self.conditional_dropout,
                  'attention_dropout': self.attention_dropout,
                  'attention_dropout2': self.attention_dropout2 if self.attend_on_both else None,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttLSTMCond2Inputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttConditionalLSTMCond2Inputs(Recurrent):
    """Long-Short Term Memory unit with the previously generated word fed to the current timestep
    and two input contexts (with two attention mechanisms).

    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        attention_mode: 'add', 'dot' or custom function.
        att_units1:  Positive integer, dimensionality of the first attention space.
        att_units2:  Positive integer, dimensionality of the second attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        attend_on_both: Boolean, wether attend on both inputs or not.
        return_states: Whether it should return the internal RNN states.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        kernel_initializer2: Initializer for the `kernel2` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer2:  Initializer for the `attention_recurrent_kernel2`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer2:  Initializer for the `attention_context_kernel2`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer2:  Initializer for the `attention_wa_kernel2`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_initializer2: Initializer for the bias vector 2
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ba_initializer2: Initializer for the bias_ba2 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer2: Initializer for the bias_ca2 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_regularizer2: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_regularizer2: Regularizer function applied to the bias2 vector
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer2:  Regularizer function applied to
            the `attention_recurrent__kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer2:  Regularizer function applied to
            the `attention_context_kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer2:  Regularizer function applied to
            the `attention_context_wa_kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer2:  Regularizer function applied to the bias_ba2 vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer2:  Regularizer function applied to the bias_ca2 vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        kernel_constraint2: Constraint function applied to
            the `kernel2` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint2: Constraint function applied to
            the `attention_recurrent_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint2: Constraint function applied to
            the `attention_context_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint2: Constraint function applied to
            the `attention_context_wa_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_constraint2: Constraint function applied to the bias2 vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ba_constraint2: Constraint function applied to
            the `bias_ba2` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint2: Constraint function applied to
            the `bias_ca2` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        dropout2: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context2.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        attention_dropout2: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism2.
        num_inputs: Number of inputs of the layer.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Egocentric Video Description based on Temporally-Linked Sequences](https://arxiv.org/abs/1704.02163)
    """

    def __init__(self, units,
                 attention_mode='add',
                 att_units1=0,
                 att_units2=0,
                 return_extra_variables=False,
                 attend_on_both=False,
                 return_states=False,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_initializer='glorot_uniform',
                 kernel_initializer2='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_recurrent_initializer2='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_initializer2='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 attention_context_wa_initializer2='glorot_uniform',
                 bias_initializer='zeros',
                 bias_initializer2='zeros',
                 bias_ba_initializer='zeros',
                 bias_ba_initializer2='zeros',
                 bias_ca_initializer='zero',
                 bias_ca_initializer2='zero',
                 kernel_regularizer=None,
                 kernel_regularizer2=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 bias_regularizer=None,
                 bias_regularizer2=None,
                 attention_recurrent_regularizer=None,
                 attention_recurrent_regularizer2=None,
                 attention_context_regularizer=None,
                 attention_context_regularizer2=None,
                 attention_context_wa_regularizer=None,
                 attention_context_wa_regularizer2=None,
                 bias_ba_regularizer=None,
                 bias_ba_regularizer2=None,
                 bias_ca_regularizer=None,
                 bias_ca_regularizer2=None,
                 kernel_constraint=None,
                 kernel_constraint2=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_recurrent_constraint2=None,
                 attention_context_constraint=None,
                 attention_context_constraint2=None,
                 attention_context_wa_constraint=None,
                 attention_context_wa_constraint2=None,
                 bias_constraint=None,
                 bias_constraint2=None,
                 bias_ba_constraint=None,
                 bias_ba_constraint2=None,
                 bias_ca_constraint=None,
                 bias_ca_constraint2=None,
                 dropout=0.,
                 dropout2=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 attention_dropout2=0.,
                 num_inputs=5,
                 **kwargs):

        super(AttConditionalLSTMCond2Inputs, self).__init__(**kwargs)

        # Main parameters
        self.units = units
        self.num_inputs = num_inputs
        self.att_units1 = units if att_units1 == 0 else att_units1
        self.att_units2 = units if att_units2 == 0 else att_units2
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attend_on_both = attend_on_both
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer2 = initializers.get(kernel_initializer2)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.recurrent_initializer_conditional = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_recurrent_initializer2 = initializers.get(attention_recurrent_initializer2)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_initializer2 = initializers.get(attention_context_initializer2)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.attention_context_wa_initializer2 = initializers.get(attention_context_wa_initializer2)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_initializer_conditional = initializers.get(bias_initializer)
        self.bias_initializer2 = initializers.get(bias_initializer2)
        self.bias_initializer2_conditional = initializers.get(bias_initializer2)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ba_initializer2 = initializers.get(bias_ba_initializer2)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.bias_ca_initializer2 = initializers.get(bias_ca_initializer2)
        self.unit_forget_bias = unit_forget_bias

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_regularizer_conditional = regularizers.get(bias_regularizer)
        self.kernel_regularizer2 = regularizers.get(kernel_regularizer2)
        self.bias_regularizer2 = regularizers.get(bias_regularizer2)
        self.bias_regularizer2_conditional = regularizers.get(bias_regularizer2)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.recurrent_regularizer_conditional = regularizers.get(recurrent_regularizer)
        # attention model learnable params
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        if self.attend_on_both:
            # attention model 2 learnable params
            self.attention_context_wa_regularizer2 = regularizers.get(attention_context_wa_regularizer2)
            self.attention_context_regularizer2 = regularizers.get(attention_context_regularizer2)
            self.attention_recurrent_regularizer2 = regularizers.get(attention_recurrent_regularizer2)
            self.bias_ba_regularizer2 = regularizers.get(bias_ba_regularizer2)
            self.bias_ca_regularizer2 = regularizers.get(bias_ca_regularizer2)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_constraint2 = constraints.get(kernel_constraint2)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.recurrent_constraint_conditional = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_recurrent_constraint2 = constraints.get(attention_recurrent_constraint2)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_constraint2 = constraints.get(attention_context_constraint2)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.attention_context_wa_constraint2 = constraints.get(attention_context_wa_constraint2)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_constraint_conditional = constraints.get(bias_constraint)
        self.bias_constraint2 = constraints.get(bias_constraint2)
        self.bias_constraint2_conditional = constraints.get(bias_constraint2)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ba_constraint2 = constraints.get(bias_ba_constraint2)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)
        self.bias_ca_constraint2 = constraints.get(bias_ca_constraint2)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.dropout2 = min(1., max(0., dropout2)) if dropout2 is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        if self.attend_on_both:
            self.attention_dropout2 = min(1., max(0., attention_dropout2)) if attention_dropout2 is not None else 0.
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3)]
        else:
            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=2)]

        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):
        assert len(input_shape) >= 3 or 'You should pass three inputs to AttConditionalLSTMCond2Inputs ' \
                                        '(previous_embedded_words, context1 and context2) and ' \
                                        'two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None, None]  # [h, c, x_att, x_att2]

        if self.attend_on_both:
            assert K.ndim(self.input_spec[1]) == 3 and K.ndim(self.input_spec[2]), 'When using two attention models,' \
                                                                                   'you should pass two 3D tensors' \
                                                                                   'to AttConditionalLSTMCond2Inputs'
        else:
            assert K.ndim(self.input_spec[1]) == 3, 'When using an attention model, you should pass one 3D tensors' \
                                                    'to AttConditionalLSTMCond2Inputs'

        if K.ndim(self.input_spec[1]) == 3:
            self.context1_steps = input_shape[1][1]
            self.context1_dim = input_shape[1][2]

        if K.ndim(self.input_spec[2]) == 3:
            self.context2_steps = input_shape[2][1]
            self.context2_dim = input_shape[2][2]
        else:
            self.context2_dim = input_shape[2][1]

        self.kernel = self.add_weight(shape=(self.context1_dim, self.units * 4),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel2 = self.add_weight(shape=(self.context2_dim, self.units * 4),
                                       initializer=self.kernel_initializer2,
                                       name='kernel2',
                                       regularizer=self.kernel_regularizer2,
                                       constraint=self.kernel_constraint2)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent_kernel_conditional = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel_conditional',
            initializer=self.recurrent_initializer_conditional,
            regularizer=self.recurrent_regularizer_conditional,
            constraint=self.recurrent_constraint_conditional)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(shape=(self.units, self.att_units1),
                                                          initializer=self.attention_recurrent_initializer,
                                                          name='attention_recurrent_kernel',
                                                          regularizer=self.attention_recurrent_regularizer,
                                                          constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(shape=(self.context1_dim, self.att_units1),
                                                        initializer=self.attention_context_initializer,
                                                        name='attention_context_kernel',
                                                        regularizer=self.attention_context_regularizer,
                                                        constraint=self.attention_context_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(shape=(self.att_units1,),
                                                        initializer=self.attention_context_wa_initializer,
                                                        name='attention_context_wa',
                                                        regularizer=self.attention_context_wa_regularizer,
                                                        constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None

        self.bias_ba = self.add_weight(shape=(self.att_units1,),
                                       initializer=self.bias_ba_initializer,
                                       name='bias_ba',
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':

            bias_ca_shape = self.context1_steps if self.context1_steps is None else (self.context1_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           initializer=self.bias_ca_initializer,
                                           name='bias_ca',
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            if self.unit_forget_bias:
                def bias_initializer_conditional(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer_conditional((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer_conditional((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer_conditional = self.bias_initializer_conditional
            self.bias_conditional = self.add_weight(shape=(self.units * 4,),
                                                    name='bias_conditional',
                                                    initializer=bias_initializer_conditional,
                                                    regularizer=self.bias_regularizer_conditional,
                                                    constraint=self.bias_constraint_conditional)
        else:
            self.bias = None
            self.bias_conditional = None
            self.bias_ba = None
            self.bias_ca = None

        if self.attend_on_both:
            # Initialize Att model params (following the same format for any option of self.consume_less)
            self.attention_recurrent_kernel2 = self.add_weight(shape=(self.units, self.att_units2),
                                                               initializer=self.attention_recurrent_initializer2,
                                                               name='attention_recurrent_kernel2',
                                                               regularizer=self.attention_recurrent_regularizer2,
                                                               constraint=self.attention_recurrent_constraint2)

            self.attention_context_kernel2 = self.add_weight(shape=(self.context2_dim, self.att_units2),
                                                             initializer=self.attention_context_initializer2,
                                                             name='attention_context_kernel2',
                                                             regularizer=self.attention_context_regularizer2,
                                                             constraint=self.attention_context_constraint2)
            if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
                self.attention_context_wa2 = self.add_weight(shape=(self.att_units2,),
                                                             initializer=self.attention_context_wa_initializer2,
                                                             name='attention_context_wa2',
                                                             regularizer=self.attention_context_wa_regularizer2,
                                                             constraint=self.attention_context_wa_constraint2)
            else:
                self.attention_context_wa2 = None
            self.bias_ba2 = self.add_weight(shape=(self.att_units2,),
                                            initializer=self.bias_ba_initializer2,
                                            name='bias_ba2',
                                            regularizer=self.bias_ba_regularizer2,
                                            constraint=self.bias_ba_constraint2)
            if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':

                bias_ca_shape = self.context2_steps if self.context2_steps is None else (self.context2_steps,)
                self.bias_ca2 = self.add_weight(shape=bias_ca_shape,
                                                initializer=self.bias_ca_initializer2,
                                                name='bias_ca2',
                                                regularizer=self.bias_ca_regularizer2,
                                                constraint=self.bias_ca_constraint2)
            else:
                self.bias_ca2

            if self.use_bias:
                if self.unit_forget_bias:
                    def bias_initializer2(shape, *args, **kwargs):
                        return K.concatenate([
                            self.bias_initializer2((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer2((self.units * 2,), *args, **kwargs),
                        ])
                else:
                    bias_initializer2 = self.bias_initializer2
                self.bias2 = self.add_weight(shape=(self.units * 4,),
                                             name='bias2',
                                             initializer=bias_initializer2,
                                             regularizer=self.bias_regularizer2,
                                             constraint=self.bias_constraint2)

                if self.unit_forget_bias:
                    def bias_initializer2_conditional(shape, *args, **kwargs):
                        return K.concatenate([
                            self.bias_initializer2_conditional((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer2_conditional((self.units * 2,), *args, **kwargs),
                        ])
                else:
                    bias_initializer2_conditional = self.bias_initializer2_conditional
                self.bias2_conditional = self.add_weight(shape=(self.units * 4,),
                                                         name='bias2_conditional',
                                                         initializer=bias_initializer2_conditional,
                                                         regularizer=self.bias_regularizer2_conditional,
                                                         constraint=self.bias_constraint2_conditional)
            else:
                self.bias2 = None
                self_bias2_conditional = None
                self.bias_ba2 = None
                self.bias_ca2 = None

        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = K.shape(self.input_spec[0][0])
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):
        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(4)]
            return K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            return K.dot(inputs, self.conditional_kernel)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context1_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            dim_x_att2 = (input_shape[0][0], input_shape[0][1], self.context2_dim)
            dim_alpha_att2 = (input_shape[0][0], input_shape[0][1], input_shape[2][1])
            main_out = [main_out, dim_x_att, dim_alpha_att, dim_x_att2, dim_alpha_att2]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim, states_dim]

        return main_out

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context1 = inputs[1]
        self.context2 = inputs[2]
        if self.num_inputs == 3:  # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 4:  # input: [state_below, context, init_generic]
            self.init_state = inputs[3]
            self.init_memory = inputs[3]
        elif self.num_inputs == 5:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[3]
            self.init_memory = inputs[4]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], mask[2], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[2, 3, 4, 5])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output
        if self.return_extra_variables:
            ret = [ret, states[2], states[3], states[4], states[5]]
        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, (list, tuple)):
                ret = [ret]
            else:
                states = list(states)
            ret += [states[0], states[1]]

        return ret

    def compute_mask(self, input, mask):

        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]
        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        pos_states = 10

        non_used_x_att = states[2]  # Placeholder for returning extra variables
        non_used_alphas_att = states[3]  # Placeholder for returning extra variables
        non_used_x_att2 = states[4]  # Placeholder for returning extra variables
        non_used_alphas_att2 = states[5]  # Placeholder for returning extra variables

        rec_dp_mask = states[6]  # Dropout U
        dp_mask2 = states[7]  # Dropout T
        dp_mask = states[8]  # Dropout W

        # Att model dropouts
        # att_dp_mask_wa = states[9]  # Dropout wa
        att_dp_mask = states[9]  # Dropout Wa
        # Att model 2 dropouts
        if self.attend_on_both:
            # att_dp_mask_wa2 = states[pos_states]  # Dropout wa
            att_dp_mask2 = states[pos_states]  # Dropout Wa

            context1 = states[pos_states + 1]  # Context
            mask_context1 = states[pos_states + 2]  # Context mask
            pctx_1 = states[pos_states + 3]  # Projected context (i.e. context * Ua + ba)

            context2 = states[pos_states + 4]  # Context 2
            mask_context2 = states[pos_states + 5]  # Context 2 mask
            pctx_2 = states[pos_states + 6]  # Projected context 2 (i.e. context * Ua2 + ba2)
        else:
            context1 = states[pos_states]  # Context
            mask_context1 = states[pos_states + 1]  # Context mask
            pctx_1 = states[pos_states + 2]  # Projected context (i.e. context * Ua + ba)

            context2 = states[pos_states + 3]  # Context 2
            mask_context2 = states[pos_states + 4]  # Context 2 mask

        if K.ndim(mask_context1) > 1:  # Mask the context (only if necessary)
            pctx_1 = mask_context1[:, :, None] * pctx_1
            context1 = mask_context1[:, :, None] * context1

        # LSTM_1
        z_ = x + K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_conditional)
        if self.use_bias:
            z_ = K.bias_add(z_, self.bias_conditional)
            if self.attend_on_both:
                z_ = K.bias_add(z_, self.bias2_conditional)
        z_0 = z_[:, :self.units]
        z_1 = z_[:, self.units: 2 * self.units]
        z_2 = z_[:, 2 * self.units: 3 * self.units]
        z_3 = z_[:, 3 * self.units:]

        i_ = self.recurrent_activation(z_0)
        f_ = self.recurrent_activation(z_1)
        c_ = f_ * c_tm1 + i_ * self.activation(z_2)
        o_ = self.recurrent_activation(z_3)
        h_ = o_ * self.activation(c_)

        # Attention model 1 (see Formulation in class header)
        ctx_, alphas = compute_attention(h_, pctx_1, context1, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, mask_context1,
                                         attention_mode=self.attention_mode)

        if self.attend_on_both:
            if K.ndim(mask_context2) > 1:  # Mask the context2 (only if necessary)
                pctx_2 = mask_context2[:, :, None] * pctx_2
                context2 = mask_context2[:, :, None] * context2
            # Attention model 2 (see Formulation in class header)
            ctx_2, alphas2 = compute_attention(h_, pctx_1, context2, att_dp_mask2, self.attention_recurrent_kernel2,
                                               self.attention_context_wa2, self.bias_ca2, mask_context2,
                                               attention_mode=self.attention_mode)
        else:
            ctx_2 = context2
            alphas2 = mask_context2

        # LSTM_2
        z = x + \
            K.dot(h_ * rec_dp_mask[0], self.recurrent_kernel) + \
            K.dot(ctx_2 * dp_mask2[0], self.kernel2) + \
            K.dot(ctx_ * dp_mask[0], self.kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
            if self.attend_on_both:
                z = K.bias_add(z, self.bias2)
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_ + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        h = o * self.activation(c)
        return h, [h, c, ctx_, alphas, ctx_2, alphas2]

    def get_constants(self, inputs, mask_context1, mask_context2, training=None):
        constants = []
        # States[6] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[7]- Dropout_T
        if 0 < self.dropout2 < 1:
            ones = K.ones_like(K.squeeze(self.context2[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout2)

            dp_mask2 = [K.in_train_phase(dropped_inputs,
                                         ones,
                                         training=training) for _ in range(4)]
            constants.append(dp_mask2)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[8]- Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context1[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # AttModel
        # States[9] - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if self.attend_on_both:
            # AttModel2
            # States[10]
            if 0 < self.attention_dropout2 < 1:
                input_dim = self.units
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * input_dim, 1)

                def dropped_inputs():
                    return K.dropout(ones, self.recurrent_dropout)

                att_dp_mask2 = [K.in_train_phase(dropped_inputs,
                                                 ones,
                                                 training=training)]
                constants.append(att_dp_mask2)
            else:
                constants.append([K.cast_to_floatx(1.)])

        # States[11] - Context1
        constants.append(self.context1)
        # States[12] - MaskContext1
        if mask_context1 is None:
            mask_context1 = K.not_equal(K.sum(self.context1, axis=2), self.mask_value)
            mask_context1 = K.cast(mask_context1, K.floatx())
        constants.append(mask_context1)

        # States[13] - pctx_1
        if 0 < self.attention_dropout < 1:
            input_dim = self.context1_dim
            ones = K.ones_like(K.reshape(self.context1[:, :, 0], (-1, K.shape(self.context1)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx_1 = K.dot(self.context1 * B_Ua[0], self.attention_context_kernel)
        else:
            pctx_1 = K.dot(self.context1, self.attention_context_kernel)
        if self.use_bias:
            pctx_1 = K.bias_add(pctx_1, self.bias_ba)
        constants.append(pctx_1)

        # States[14] - Context2
        constants.append(self.context2)
        # States[15] - MaskContext2
        if self.attend_on_both:
            if mask_context2 is None:
                mask_context2 = K.not_equal(K.sum(self.context2, axis=2), self.mask_value)
                mask_context2 = K.cast(mask_context2, K.floatx())
        else:
            mask_context2 = K.ones_like(self.context2[:, 0])
        constants.append(mask_context2)
        if self.attend_on_both:
            # States[16] - pctx_2
            if 0 < self.attention_dropout2 < 1:
                input_dim = self.context2_dim
                ones = K.ones_like(K.reshape(self.context2[:, :, 0], (-1, K.shape(self.context2)[1], 1)))
                ones = K.concatenate([ones] * input_dim, axis=2)
                B_Ua2 = [K.in_train_phase(K.dropout(ones, self.attention_dropout2), ones)]
                pctx_2 = K.dot(self.context2 * B_Ua2[0], self.attention_context_kernel2)
            else:
                pctx_2 = K.dot(self.context2, self.attention_context_kernel2)
            if self.use_bias:
                pctx_2 = K.bias_add(pctx_2, self.bias_ba2)
            constants.append(pctx_2)

        return constants

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            # build an all-zero tensor of shape (samples, units)
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        # extra states for context1 and context2
        initial_state1 = K.zeros_like(self.context1)  # (samples, input_timesteps, ctx1_dim)
        initial_state_alphas1 = K.sum(initial_state1, axis=2)  # (samples, input_timesteps)
        initial_state1 = K.sum(initial_state1, axis=1)  # (samples, ctx1_dim)
        extra_states = [initial_state1, initial_state_alphas1]
        initial_state2 = K.zeros_like(self.context2)  # (samples, input_timesteps, ctx2_dim)
        if self.attend_on_both:  # Reduce on temporal dimension
            initial_state_alphas2 = K.sum(initial_state2, axis=2)  # (samples, input_timesteps)
            initial_state2 = K.sum(initial_state2, axis=1)  # (samples, ctx2_dim)
        else:  # Already reduced
            initial_state_alphas2 = initial_state2  # (samples, ctx2_dim)

        extra_states.append(initial_state2)
        extra_states.append(initial_state_alphas2)

        return initial_states + extra_states

    def get_config(self):
        config = {"units": self.units,
                  "att_units1": self.att_units1,
                  "att_units2": self.att_units2,
                  "return_extra_variables": self.return_extra_variables,
                  "return_states": self.return_states,
                  "use_bias": self.use_bias,
                  "mask_value": self.mask_value,
                  "attend_on_both": self.attend_on_both,
                  'unit_forget_bias': self.unit_forget_bias,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                  "kernel_regularizer2": regularizers.serialize(self.kernel_regularizer2),
                  "conditional_regularizer": regularizers.serialize(self.conditional_regularizer),
                  "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
                  "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                  'attention_context_wa_regularizer': regularizers.serialize(self.attention_context_wa_regularizer),
                  'attention_context_regularizer': regularizers.serialize(self.attention_context_regularizer),
                  'attention_recurrent_regularizer': regularizers.serialize(self.attention_recurrent_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  "bias_regularizer2": regularizers.serialize(self.bias_regularizer2),
                  'attention_context_wa_regularizer2': regularizers.serialize(self.attention_context_wa_regularizer2),
                  'attention_context_regularizer2': regularizers.serialize(self.attention_context_regularizer2),
                  'attention_recurrent_regularizer2': regularizers.serialize(self.attention_recurrent_regularizer2),
                  'bias_ba_regularizer2': regularizers.serialize(self.bias_ba_regularizer2),
                  'bias_ca_regularizer2': regularizers.serialize(self.bias_ca_regularizer2),
                  "kernel_initializer": initializers.serialize(self.kernel_initializer),
                  "kernel_initializer2": initializers.serialize(self.kernel_initializer2),
                  "conditional_initializer": initializers.serialize(self.conditional_initializer),
                  "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
                  "bias_initializer": initializers.serialize(self.bias_initializer),
                  'attention_context_wa_initializer': initializers.serialize(self.attention_context_wa_initializer),
                  'attention_context_initializer': initializers.serialize(self.attention_context_initializer),
                  'attention_recurrent_initializer': initializers.serialize(self.attention_recurrent_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  "bias_initializer2": initializers.serialize(self.bias_initializer2),
                  'attention_context_wa_initializer2': initializers.serialize(self.attention_context_wa_initializer2),
                  'attention_context_initializer2': initializers.serialize(self.attention_context_initializer2),
                  'attention_recurrent_initializer2': initializers.serialize(self.attention_recurrent_initializer2),
                  'bias_ba_initializer2': initializers.serialize(self.bias_ba_initializer2),
                  'bias_ca_initializer2': initializers.serialize(self.bias_ca_initializer2),
                  "kernel_constraint": constraints.serialize(self.kernel_constraint),
                  "kernel_constraint2": constraints.serialize(self.kernel_constraint2),
                  "conditional_constraint": constraints.serialize(self.conditional_constraint),
                  "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
                  "bias_constraint": constraints.serialize(self.bias_constraint),
                  'attention_context_wa_constraint': constraints.serialize(self.attention_context_wa_constraint),
                  'attention_context_constraint': constraints.serialize(self.attention_context_constraint),
                  'attention_recurrent_constraint': constraints.serialize(self.attention_recurrent_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
                  "bias_constraint2": constraints.serialize(self.bias_constraint2),
                  'attention_context_wa_constraint2': constraints.serialize(self.attention_context_wa_constraint2),
                  'attention_context_constraint2': constraints.serialize(self.attention_context_constraint2),
                  'attention_recurrent_constraint2': constraints.serialize(self.attention_recurrent_constraint2),
                  'bias_ba_constraint2': constraints.serialize(self.bias_ba_constraint2),
                  'bias_ca_constraint2': constraints.serialize(self.bias_ca_constraint2),
                  "dropout": self.dropout,
                  "dropout2": self.dropout2,
                  "recurrent_dropout": self.recurrent_dropout,
                  "conditional_dropout": self.conditional_dropout,
                  'attention_dropout': self.attention_dropout,
                  'attention_dropout2': self.attention_dropout2 if self.attend_on_both else None,
                  'attention_mode': self.attention_mode
                  }
        base_config = super(AttConditionalLSTMCond2Inputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttLSTMCond3Inputs(Recurrent):
    """Long-Short Term Memory unit with the previously generated word fed to the current timestep
    and three input contexts (with three attention mechanisms).

    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units1:  Positive integer, dimensionality of the first attention space.
        att_units2:  Positive integer, dimensionality of the second attention space.
        att_units3:  Positive integer, dimensionality of the third attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        attend_on_both: Boolean, wether attend on both inputs or not.
        return_states: Whether it should return the internal RNN states.
        attention_mode: 'add', 'dot' or custom function.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        kernel_initializer2: Initializer for the `kernel2` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        kernel_initializer3: Initializer for the `kernel3` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer2:  Initializer for the `attention_recurrent_kernel2`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer3:  Initializer for the `attention_recurrent_kernel3`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer2:  Initializer for the `attention_context_kernel2`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer3:  Initializer for the `attention_context_kernel3`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer2:  Initializer for the `attention_wa_kernel2`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer3:  Initializer for the `attention_wa_kernel3`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_initializer2: Initializer for the bias vector 2
            (see [initializers](../initializers.md)).
        bias_initializer3: Initializer for the bias vector 3
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ba_initializer2: Initializer for the bias_ba2 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ba_initializer3: Initializer for the bias_ba3 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer2: Initializer for the bias_ca2 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer3: Initializer for the bias_ca3 vector from the attention mechanism
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_regularizer2: Regularizer function applied to
            the `kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_regularizer3: Regularizer function applied to
            the `kernel3` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_regularizer2: Regularizer function applied to the bias2 vector
            (see [regularizer](../regularizers.md)).
        bias_regularizer3: Regularizer function applied to the bias3 vector
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer2:  Regularizer function applied to
            the `attention_recurrent__kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer3:  Regularizer function applied to
            the `attention_recurrent__kernel3` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer2:  Regularizer function applied to
            the `attention_context_kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer3:  Regularizer function applied to
            the `attention_context_kernel3` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer2:  Regularizer function applied to
            the `attention_context_wa_kernel2` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer3:  Regularizer function applied to
            the `attention_context_wa_kernel3` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer2:  Regularizer function applied to the bias_ba2 vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer3:  Regularizer function applied to the bias_ba3 vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer2:  Regularizer function applied to the bias_ca2 vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer3:  Regularizer function applied to the bias_ca3 vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        kernel_constraint2: Constraint function applied to
            the `kernel2` weights matrix
            (see [constraints](../constraints.md)).
        kernel_constraint3: Constraint function applied to
            the `kernel3` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint2: Constraint function applied to
            the `attention_recurrent_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint3: Constraint function applied to
            the `attention_recurrent_kernel3` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint2: Constraint function applied to
            the `attention_context_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint3: Constraint function applied to
            the `attention_context_kernel3` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint2: Constraint function applied to
            the `attention_context_wa_kernel2` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint3: Constraint function applied to
            the `attention_context_wa_kernel3` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_constraint2: Constraint function applied to the bias2 vector
            (see [constraints](../constraints.md)).
        bias_constraint3: Constraint function applied to the bias3 vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ba_constraint2: Constraint function applied to
            the `bias_ba2` weights matrix
            (see [constraints](../constraints.md)).
        bias_ba_constraint3: Constraint function applied to
            the `bias_ba3` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint2: Constraint function applied to
            the `bias_ca2` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint3: Constraint function applied to
            the `bias_ca3` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        dropout2: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context2.
        dropout3: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context3.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        attention_dropout2: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism2.
        attention_dropout3: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism3.
        num_inputs: Number of inputs of the layer.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Egocentric Video Description based on Temporally-Linked Sequences](https://arxiv.org/abs/1704.02163)
    """

    def __init__(self,
                 units,
                 att_units1=0,
                 att_units2=0,
                 att_units3=0,
                 return_extra_variables=False,
                 attend_on_both=False,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_initializer='glorot_uniform',
                 kernel_initializer2='glorot_uniform',
                 kernel_initializer3='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_recurrent_initializer2='glorot_uniform',
                 attention_recurrent_initializer3='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_initializer2='glorot_uniform',
                 attention_context_initializer3='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 attention_context_wa_initializer2='glorot_uniform',
                 attention_context_wa_initializer3='glorot_uniform',
                 bias_initializer='zeros',
                 bias_initializer2='zeros',
                 bias_initializer3='zeros',
                 bias_ba_initializer='zeros',
                 bias_ba_initializer2='zeros',
                 bias_ba_initializer3='zeros',
                 bias_ca_initializer='zero',
                 bias_ca_initializer2='zero',
                 bias_ca_initializer3='zero',
                 kernel_regularizer=None,
                 kernel_regularizer2=None,
                 kernel_regularizer3=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 bias_regularizer=None,
                 bias_regularizer2=None,
                 bias_regularizer3=None,
                 attention_recurrent_regularizer=None,
                 attention_recurrent_regularizer2=None,
                 attention_recurrent_regularizer3=None,
                 attention_context_regularizer=None,
                 attention_context_regularizer2=None,
                 attention_context_regularizer3=None,
                 attention_context_wa_regularizer=None,
                 attention_context_wa_regularizer2=None,
                 attention_context_wa_regularizer3=None,
                 bias_ba_regularizer=None,
                 bias_ba_regularizer2=None,
                 bias_ba_regularizer3=None,
                 bias_ca_regularizer=None,
                 bias_ca_regularizer2=None,
                 bias_ca_regularizer3=None,
                 kernel_constraint=None,
                 kernel_constraint2=None,
                 kernel_constraint3=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_recurrent_constraint2=None,
                 attention_recurrent_constraint3=None,
                 attention_context_constraint=None,
                 attention_context_constraint2=None,
                 attention_context_constraint3=None,
                 attention_context_wa_constraint=None,
                 attention_context_wa_constraint2=None,
                 attention_context_wa_constraint3=None,
                 bias_constraint=None,
                 bias_constraint2=None,
                 bias_constraint3=None,
                 bias_ba_constraint=None,
                 bias_ba_constraint2=None,
                 bias_ba_constraint3=None,
                 bias_ca_constraint=None,
                 bias_ca_constraint2=None,
                 bias_ca_constraint3=None,
                 dropout=0.,
                 dropout2=0.,
                 dropout3=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 attention_dropout2=0.,
                 attention_dropout3=0.,
                 num_inputs=6,
                 **kwargs):

        super(AttLSTMCond3Inputs, self).__init__(**kwargs)

        # Main parameters
        self.units = units
        self.num_inputs = num_inputs
        self.att_units1 = units if att_units1 == 0 else att_units1
        self.att_units2 = units if att_units2 == 0 else att_units2
        self.att_units3 = units if att_units3 == 0 else att_units3
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attend_on_both = attend_on_both
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states
        self.attention_mode = attention_mode.lower()

        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer2 = initializers.get(kernel_initializer2)
        self.kernel_initializer3 = initializers.get(kernel_initializer3)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_recurrent_initializer2 = initializers.get(attention_recurrent_initializer2)
        self.attention_recurrent_initializer3 = initializers.get(attention_recurrent_initializer3)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_initializer2 = initializers.get(attention_context_initializer2)
        self.attention_context_initializer3 = initializers.get(attention_context_initializer3)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.attention_context_wa_initializer2 = initializers.get(attention_context_wa_initializer2)
        self.attention_context_wa_initializer3 = initializers.get(attention_context_wa_initializer3)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_initializer2 = initializers.get(bias_initializer2)
        self.bias_initializer3 = initializers.get(bias_initializer3)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ba_initializer2 = initializers.get(bias_ba_initializer2)
        self.bias_ba_initializer3 = initializers.get(bias_ba_initializer3)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.bias_ca_initializer2 = initializers.get(bias_ca_initializer2)
        self.bias_ca_initializer3 = initializers.get(bias_ca_initializer3)
        self.unit_forget_bias = unit_forget_bias

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_regularizer2 = regularizers.get(kernel_regularizer2)
        self.kernel_regularizer3 = regularizers.get(kernel_regularizer3)
        self.bias_regularizer2 = regularizers.get(bias_regularizer2)
        self.bias_regularizer3 = regularizers.get(bias_regularizer3)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        # attention model learnable params
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        if self.attend_on_both:
            # attention model 2 learnable params
            self.attention_context_wa_regularizer2 = regularizers.get(attention_context_wa_regularizer2)
            self.attention_context_regularizer2 = regularizers.get(attention_context_regularizer2)
            self.attention_recurrent_regularizer2 = regularizers.get(attention_recurrent_regularizer2)
            self.bias_ba_regularizer2 = regularizers.get(bias_ba_regularizer2)
            self.bias_ca_regularizer2 = regularizers.get(bias_ca_regularizer2)
            # attention model 3 learnable params
            self.attention_context_wa_regularize3 = regularizers.get(attention_context_wa_regularizer3)
            self.attention_context_regularizer3 = regularizers.get(attention_context_regularizer3)
            self.attention_recurrent_regularizer3 = regularizers.get(attention_recurrent_regularizer3)
            self.bias_ba_regularizer3 = regularizers.get(bias_ba_regularizer3)
            self.bias_ca_regularizer3 = regularizers.get(bias_ca_regularizer3)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_constraint2 = constraints.get(kernel_constraint2)
        self.kernel_constraint3 = constraints.get(kernel_constraint3)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_recurrent_constraint2 = constraints.get(attention_recurrent_constraint2)
        self.attention_recurrent_constraint3 = constraints.get(attention_recurrent_constraint3)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_constraint2 = constraints.get(attention_context_constraint2)
        self.attention_context_constraint3 = constraints.get(attention_context_constraint3)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.attention_context_wa_constraint2 = constraints.get(attention_context_wa_constraint2)
        self.attention_context_wa_constraint3 = constraints.get(attention_context_wa_constraint3)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_constraint2 = constraints.get(bias_constraint2)
        self.bias_constraint3 = constraints.get(bias_constraint3)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ba_constraint2 = constraints.get(bias_ba_constraint2)
        self.bias_ba_constraint3 = constraints.get(bias_ba_constraint3)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)
        self.bias_ca_constraint2 = constraints.get(bias_ca_constraint2)
        self.bias_ca_constraint3 = constraints.get(bias_ca_constraint3)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.dropout2 = min(1., max(0., dropout2)) if dropout2 is not None else 0.
        self.dropout3 = min(1., max(0., dropout3)) if dropout3 is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        if self.attend_on_both:
            self.attention_dropout2 = min(1., max(0., attention_dropout2)) if attention_dropout2 is not None else 0.
            self.attention_dropout3 = min(1., max(0., attention_dropout3)) if attention_dropout3 is not None else 0.

        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):
        assert len(input_shape) >= 4 or 'You should pass four inputs to AttLSTMCond3Inputs ' \
                                        '(previous_embedded_words, context1, context2, context3) and ' \
                                        'two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None, None, None]  # [h, c, x_att, x_att2, x_att3]

        if self.attend_on_both:
            assert K.ndim(self.input_spec[1]) == 3 and K.ndim(self.input_spec[2]) and K.ndim(self.input_spec[3]), 'When using three attention models,' \
                                                                                                                  'you should pass three 3D tensors' \
                                                                                                                  'to AttLSTMCond3Inputs'
        else:
            assert self.input_spec[1].ndim == 3, 'When using an attention model, you should pass one 3D tensors' \
                                                 'to AttLSTMCond3Inputs'

        if K.ndim(self.input_spec[1]) == 3:
            self.context1_steps = input_shape[1][1]
            self.context1_dim = input_shape[1][2]

        if K.ndim(self.input_spec[2]) == 3:
            self.context2_steps = input_shape[2][1]
            self.context2_dim = input_shape[2][2]
        else:
            self.context2_dim = input_shape[2][1]

        if K.ndim(self.input_spec[3]) == 3:
            self.context3_steps = input_shape[3][1]
            self.context3_dim = input_shape[3][2]
        else:
            self.context3_dim = input_shape[3][1]

        # Initialize Att model params
        self.kernel = self.add_weight(shape=(self.context1_dim, self.units * 4),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel2 = self.add_weight(shape=(self.context2_dim, self.units * 4),
                                       initializer=self.kernel_initializer2,
                                       name='kernel2',
                                       regularizer=self.kernel_regularizer2,
                                       constraint=self.kernel_constraint2)

        self.kernel3 = self.add_weight(shape=(self.context3_dim, self.units * 4),
                                       initializer=self.kernel_initializer3,
                                       name='kernel3',
                                       regularizer=self.kernel_regularizer3,
                                       constraint=self.kernel_constraint3)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(shape=(self.units, self.att_units1),
                                                          initializer=self.attention_recurrent_initializer,
                                                          name='attention_recurrent_kernel',
                                                          regularizer=self.attention_recurrent_regularizer,
                                                          constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(shape=(self.context1_dim, self.att_units1),
                                                        initializer=self.attention_context_initializer,
                                                        name='attention_context_kernel',
                                                        regularizer=self.attention_context_regularizer,
                                                        constraint=self.attention_context_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(shape=(self.att_units1,),
                                                        initializer=self.attention_context_wa_initializer,
                                                        name='attention_context_wa',
                                                        regularizer=self.attention_context_wa_regularizer,
                                                        constraint=self.attention_context_wa_constraint)
        else:
            self.attention_context_wa = None

        self.bias_ba = self.add_weight(shape=(self.att_units1,),
                                       initializer=self.bias_ba_initializer,
                                       name='bias_ba',
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            bias_ca_shape = self.context1_steps if self.context1_steps is None else (self.context1_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           initializer=self.bias_ca_initializer,
                                           name='bias_ca',
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.attend_on_both and self.attention_mode == 'add' or self.attention_mode == 'bahdanau':

            # Initialize Att model params (following the same format for any option of self.consume_less)
            self.wa2 = self.add_weight((self.att_units2,),
                                       initializer=self.init,
                                       name='{}_wa2'.format(self.name),
                                       regularizer=self.wa2_regularizer)

            self.Wa2 = self.add_weight((self.units, self.att_units2),
                                       initializer=self.init,
                                       name='{}_Wa2'.format(self.name),
                                       regularizer=self.Wa2_regularizer)
            self.Ua2 = self.add_weight((self.context2_dim, self.att_units2),
                                       initializer=self.inner_init,
                                       name='{}_Ua2'.format(self.name),
                                       regularizer=self.Ua2_regularizer)

            self.ba2 = self.add_weight(shape=self.att_units2,
                                       name='{}_ba2'.format(self.name),
                                       initializer='zero',
                                       regularizer=self.ba2_regularizer)

            self.ca2 = self.add_weight(shape=self.context2_steps,
                                       name='{}_ca2'.format(self.name),
                                       initializer='zero',
                                       regularizer=self.ca2_regularizer)

            self.wa3 = self.add_weight(shape=(self.att_units3,),
                                       initializer=self.init,
                                       name='{}_wa3'.format(self.name),
                                       regularizer=self.wa3_regularizer)

            self.Wa3 = self.add_weight(shape=(self.units, self.att_units3),
                                       initializer=self.init,
                                       name='{}_Wa3'.format(self.name),
                                       regularizer=self.Wa3_regularizer)
            self.Ua3 = self.add_weight(shape=(self.context3_dim, self.att_units3),
                                       initializer=self.inner_init,
                                       name='{}_Ua3'.format(self.name),
                                       regularizer=self.Ua3_regularizer)

            self.ba3 = self.add_weight(shape=self.att_units3,
                                       name='{}_ba3'.format(self.name),
                                       initializer='zero',
                                       regularizer=self.ba3_regularizer)

            self.ca3 = self.add_weight(shape=self.context3_steps,
                                       name='{}_ca3'.format(self.name),
                                       initializer='zero',
                                       regularizer=self.ca3_regularizer)
        else:
            self.wa2 = None
            self.Wa2 = None
            self.Ua2 = None
            self.ba2 = None
            self.ca2 = None
            self.wa3 = None
            self.Wa3 = None
            self.Ua3 = None
            self.ba3 = None
            self.ca3 = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = K.shape(self.input_spec[0][0])
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, x, B_V):
        return K.dot(x * B_V[0], self.V)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context1_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            dim_x_att2 = (input_shape[0][0], input_shape[0][1], self.context2_dim)
            dim_alpha_att2 = (input_shape[0][0], input_shape[0][1], input_shape[2][1])
            dim_x_att3 = (input_shape[0][0], input_shape[0][1], self.context3_dim)
            dim_alpha_att3 = (input_shape[0][0], input_shape[0][1], input_shape[3][1])

            main_out = [main_out, dim_x_att, dim_alpha_att,
                        dim_x_att2, dim_alpha_att2,
                        dim_x_att3, dim_alpha_att3]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim, states_dim]

        return main_out

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.

        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context1 = x[1]
        self.context2 = x[2]
        self.context3 = x[3]

        if self.num_inputs == 4:  # input: [state_below, context, context3]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 5:  # input: [state_below, context, context2, init_generic]
            self.init_state = x[4]
            self.init_memory = x[4]
        elif self.num_inputs == 6:  # input: [state_below, context, context2,  init_state, init_memory]
            self.init_state = x[4]
            self.init_memory = x[5]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants, B_V = self.get_constants(state_below, mask[1], mask[2], mask[3])

        preprocessed_input = self.preprocess_input(state_below, B_V)

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[2, 3, 4, 5, 6, 7])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output
        if self.return_extra_variables:
            ret = [ret, states[2], states[3], states[4], states[5], states[6], states[7]]
        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0], states[1]]

        return ret

    def compute_mask(self, input, mask):

        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0], mask[0], mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]
        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        pos_states = 14

        non_used_x_att = states[2]  # Placeholder for returning extra variables
        non_used_alphas_att = states[3]  # Placeholder for returning extra variables

        non_used_x_att2 = states[4]  # Placeholder for returning extra variables
        non_used_alphas_att2 = states[5]  # Placeholder for returning extra variables

        non_used_x_att3 = states[6]  # Placeholder for returning extra variables
        non_used_alphas_att3 = states[7]  # Placeholder for returning extra variables

        B_U = states[8]  # Dropout U
        B_T = states[9]  # Dropout T
        B_W = states[10]  # Dropout W
        B_S = states[11]  # Dropout T

        # Att model dropouts
        B_wa = states[12]  # Dropout wa
        B_Wa = states[13]  # Dropout Wa
        # Att model 2 dropouts
        if self.attend_on_both:
            B_wa2 = states[pos_states]  # Dropout wa
            B_Wa2 = states[pos_states + 1]  # Dropout Wa
            B_wa3 = states[pos_states + 2]  # Dropout wa3
            B_Wa3 = states[pos_states + 3]  # Dropout Wa3

            context1 = states[pos_states + 4]  # Context
            mask_context1 = states[pos_states + 5]  # Context mask
            pctx_1 = states[pos_states + 6]  # Projected context (i.e. context * Ua + ba)

            context2 = states[pos_states + 7]  # Context 2
            mask_context2 = states[pos_states + 8]  # Context 2 mask
            pctx_2 = states[pos_states + 9]  # Projected context 2 (i.e. context * Ua2 + ba2)

            context3 = states[pos_states + 10]  # Context 3
            mask_context3 = states[pos_states + 11]  # Context 3 mask
            pctx_3 = states[pos_states + 12]  # Projected context 3 (i.e. context * Ua3 + ba3)

        else:
            context1 = states[pos_states]  # Context
            mask_context1 = states[pos_states + 1]  # Context mask
            pctx_1 = states[pos_states + 2]  # Projected context (i.e. context * Ua + ba)

            context2 = states[pos_states + 3]  # Context 2
            mask_context2 = states[pos_states + 4]  # Context 2 mask

            context3 = states[pos_states + 5]  # Context 2
            mask_context3 = states[pos_states + 6]  # Context 2 mask

        if K.ndim(mask_context1) > 1:  # Mask the context (only if necessary)
            pctx_1 = mask_context1[:, :, None] * pctx_1
            context1 = mask_context1[:, :, None] * context1

        # Attention model 1 (see Formulation in class header)
        p_state_1 = K.dot(h_tm1 * B_Wa[0], self.Wa)
        pctx_1 = K.tanh(pctx_1 + p_state_1[:, None, :])
        e1 = K.dot(pctx_1 * B_wa[0], self.wa) + self.ca
        if K.ndim(mask_context1) > 1:  # Mask the context (only if necessary)
            e1 = mask_context1 * e1
        alphas1 = K.softmax(e1.reshape([K.shape(e1)[0], K.shape(e1)[1]]))
        # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        ctx_1 = K.sum(context1 * alphas1[:, :, None], axis=1)

        if self.attend_on_both:
            if K.ndim(mask_context2) > 1:  # Mask the context2 (only if necessary)
                pctx_2 = mask_context2[:, :, None] * pctx_2
                context2 = mask_context2[:, :, None] * context2
            if K.ndim(mask_context3) > 1:  # Mask the context2 (only if necessary)
                pctx_3 = mask_context3[:, :, None] * pctx_3
                context3 = mask_context3[:, :, None] * context3

        if self.attend_on_both:
            # Attention model 2 (see Formulation in class header)
            ctx_2, alphas2 = compute_attention(h_tm1, pctx_2, context2, B_Wa2, self.Wa2,
                                               self.wa2, self.ca2, mask_context2,
                                               attention_mode=self.attention_mode)
            # Attention model 3 (see Formulation in class header)
            ctx_3, alphas3 = compute_attention(h_tm1, pctx_3, context3, B_Wa3, self.Wa3,
                                               self.wa3, self.ca3, mask_context3,
                                               attention_mode=self.attention_mode)
        else:
            ctx_2 = context2
            alphas2 = mask_context2
            ctx_3 = context3
            alphas3 = mask_context3

        z = x + \
            K.dot(h_tm1 * B_U[0], self.U) + \
            K.dot(ctx_1 * B_T[0], self.T) + \
            K.dot(ctx_2 * B_W[0], self.W) + \
            K.dot(ctx_3 * B_S[0], self.S) + \
            self.b
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3)
        h = o * self.activation(c)

        return h, [h, c, ctx_1, alphas1, ctx_2, alphas2, ctx_3, alphas3]

    def get_constants(self, x, mask_context1, mask_context2, mask_context3):
        constants = []
        # States[8]
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.units, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[9]
        if 0 < self.dropout_T < 1:
            input_shape = K.shape(self.input_spec[1][0])
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_T = [K.in_train_phase(K.dropout(ones, self.dropout_T), ones) for _ in range(4)]
            constants.append(B_T)
        else:
            B_T = [K.cast_to_floatx(1.) for _ in range(4)]
        constants.append(B_T)

        # States[10]
        if 0 < self.dropout_W < 1:
            input_shape = K.shape(self.input_spec[2][0])
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            B_W = [K.cast_to_floatx(1.) for _ in range(4)]
        constants.append(B_W)

        # States[11]
        if 0 < self.dropout_S < 1:
            input_shape = K.shape(self.input_spec[3][0])
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_S = [K.in_train_phase(K.dropout(ones, self.dropout_S), ones) for _ in range(4)]
            constants.append(B_S)
        else:
            B_S = [K.cast_to_floatx(1.) for _ in range(4)]
        constants.append(B_S)

        # AttModel
        # States[12]
        if 0 < self.dropout_wa < 1:
            ones = K.ones_like(K.reshape(self.context1[:, :, 0], (-1, K.shape(self.context1)[1], 1)))
            # ones = K.concatenate([ones], 1)
            B_wa = [K.in_train_phase(K.dropout(ones, self.dropout_wa), ones)]
            constants.append(B_wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        # States[13]
        if 0 < self.dropout_Wa < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_Wa = [K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)]
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if self.attend_on_both:
            # AttModel2
            # States[14]
            if 0 < self.dropout_wa2 < 1:
                ones = K.ones_like(K.reshape(self.context2[:, :, 0], (-1, K.shape(self.context2)[1], 1)))
                # ones = K.concatenate([ones], 1)
                B_wa2 = [K.in_train_phase(K.dropout(ones, self.dropout_wa2), ones)]
                constants.append(B_wa2)
            else:
                constants.append([K.cast_to_floatx(1.)])

            # States[15]
            if 0 < self.dropout_Wa2 < 1:
                input_dim = self.units
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * input_dim, 1)
                B_Wa2 = [K.in_train_phase(K.dropout(ones, self.dropout_Wa2), ones)]
                constants.append(B_Wa2)
            else:
                constants.append([K.cast_to_floatx(1.)])

            # States[16]
            if 0 < self.dropout_wa3 < 1:
                ones = K.ones_like(K.reshape(self.context2[:, :, 0], (-1, K.shape(self.context3)[1], 1)))
                B_wa3 = [K.in_train_phase(K.dropout(ones, self.dropout_wa3), ones)]
                constants.append(B_wa3)
            else:
                constants.append([K.cast_to_floatx(1.)])

            # States[17]
            if 0 < self.dropout_Wa3 < 1:
                input_dim = self.units
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * input_dim, 1)
                B_Wa3 = [K.in_train_phase(K.dropout(ones, self.dropout_Wa3), ones)]
                constants.append(B_Wa3)
            else:
                constants.append([K.cast_to_floatx(1.)])

        # States[18] - [14]
        constants.append(self.context1)
        # States [19] - [15]
        if mask_context1 is None:
            mask_context1 = K.not_equal(K.sum(self.context1, axis=2), self.mask_value)
        constants.append(mask_context1)

        # States [20] - [15]
        if 0 < self.dropout_Ua < 1:
            input_dim = self.context1_dim
            ones = K.ones_like(K.reshape(self.context1[:, :, 0], (-1, K.shape(self.context1)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.dropout_Ua), ones)]
            pctx1 = K.dot(self.context1 * B_Ua[0], self.Ua) + self.ba
        else:
            pctx1 = K.dot(self.context1, self.Ua) + self.ba
        constants.append(pctx1)

        # States[21] - [16]
        constants.append(self.context2)
        # States [22] - [17]
        if self.attend_on_both:
            if mask_context2 is None:
                mask_context2 = K.not_equal(K.sum(self.context2, axis=2), self.mask_value)
        else:
            mask_context2 = K.ones_like(self.context2[:, 0])
        constants.append(mask_context2)

        # States [23] - [18]
        if self.attend_on_both:
            if 0 < self.dropout_Ua2 < 1:
                input_dim = self.context2_dim
                ones = K.ones_like(K.reshape(self.context2[:, :, 0], (-1, K.shape(self.context2)[1], 1)))
                ones = K.concatenate([ones] * input_dim, axis=2)
                B_Ua2 = [K.in_train_phase(K.dropout(ones, self.dropout_Ua2), ones)]
                pctx2 = K.dot(self.context2 * B_Ua2[0], self.Ua2) + self.ba2
            else:
                pctx2 = K.dot(self.context2, self.Ua2) + self.ba2
            constants.append(pctx2)

        # States[24] - [19]
        constants.append(self.context3)
        # States [25] - [20]
        if self.attend_on_both:
            if mask_context3 is None:
                mask_context3 = K.not_equal(K.sum(self.context3, axis=2), self.mask_value)
        else:
            mask_context3 = K.ones_like(self.context3[:, 0])
        constants.append(mask_context3)

        # States [26] - [21]
        if self.attend_on_both:
            if 0 < self.dropout_Ua3 < 1:
                input_dim = self.context3_dim
                ones = K.ones_like(K.reshape(self.context3[:, :, 0], (-1, K.shape(self.context3)[1], 1)))
                ones = K.concatenate([ones] * input_dim, axis=2)
                B_Ua3 = [K.in_train_phase(K.dropout(ones, self.dropout_Ua3), ones)]
                pctx3 = K.dot(self.context3 * B_Ua3[0], self.Ua3) + self.ba3
            else:
                pctx3 = K.dot(self.context3, self.Ua3) + self.ba3
            constants.append(pctx3)

        if 0 < self.dropout_V < 1:
            input_dim = self.input_dim
            ones = K.ones_like(K.reshape(x[:, :, 0], (-1, K.shape(x)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_V = [K.in_train_phase(K.dropout(ones, self.dropout_V), ones) for _ in range(4)]
        else:
            B_V = [K.cast_to_floatx(1.) for _ in range(4)]
        return constants, B_V

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            # build an all-zero tensor of shape (samples, units)
            initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        # extra states for context1 and context2 and context3
        initial_state1 = K.zeros_like(self.context1)  # (samples, input_timesteps, ctx1_dim)
        initial_state_alphas1 = K.sum(initial_state1, axis=2)  # (samples, input_timesteps)
        initial_state1 = K.sum(initial_state1, axis=1)  # (samples, ctx1_dim)
        extra_states = [initial_state1, initial_state_alphas1]
        initial_state2 = K.zeros_like(self.context2)  # (samples, input_timesteps, ctx2_dim)
        initial_state3 = K.zeros_like(self.context3)  # (samples, input_timesteps, ctx2_dim)

        if self.attend_on_both:  # Reduce on temporal dimension
            initial_state_alphas2 = K.sum(initial_state2, axis=2)  # (samples, input_timesteps)
            initial_state2 = K.sum(initial_state2, axis=1)  # (samples, ctx2_dim)
            initial_state_alphas3 = K.sum(initial_state3, axis=2)  # (samples, input_timesteps)
            initial_state3 = K.sum(initial_state3, axis=1)  # (samples, ctx3_dim)
        else:  # Already reduced
            initial_state_alphas2 = initial_state2  # (samples, ctx2_dim)
            initial_state_alphas3 = initial_state3  # (samples, ctx2_dim)

        extra_states.append(initial_state2)
        extra_states.append(initial_state_alphas2)

        extra_states.append(initial_state3)
        extra_states.append(initial_state_alphas3)
        return initial_states + extra_states

    def get_config(self):
        config = {"units": self.units,
                  "att_units1": self.att_units1,
                  "att_units2": self.att_units2,
                  "att_units3": self.att_units3,
                  "return_extra_variables": self.return_extra_variables,
                  "return_states": self.return_states,
                  "mask_value": self.mask_value,
                  "attend_on_both": self.attend_on_both,
                  "kernel_initializer": initializers.serialize(self.W_regularizer),
                  "recurrent_initializer": initializers.serialize(self.U_regularizer),
                  "unit_forget_bias": initializers.serialize(self.forget_bias_init),
                  "activation": activations.serialize(self.activation),
                  'attention_mode': self.attention_mode,
                  'use_bias': self.use_bias,
                  "recurrent_activation": activations.serialize(self.inner_activation),
                  "S_regularizer": self.S_regularizer.get_config() if self.S_regularizer else None,
                  "T_regularizer": self.T_regularizer.get_config() if self.T_regularizer else None,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "V_regularizer": self.V_regularizer.get_config() if self.V_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  'wa_regularizer': self.wa_regularizer.get_config() if self.wa_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'Ua_regularizer': self.Ua_regularizer.get_config() if self.Ua_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'ca_regularizer': self.ca_regularizer.get_config() if self.ca_regularizer else None,
                  'wa2_regularizer': self.wa2_regularizer.get_config() if self.attend_on_both and self.wa2_regularizer else None,
                  'Wa2_regularizer': self.Wa2_regularizer.get_config() if self.attend_on_both and self.Wa2_regularizer else None,
                  'Ua2_regularizer': self.Ua2_regularizer.get_config() if self.attend_on_both and self.Ua2_regularizer else None,
                  'ba2_regularizer': self.ba2_regularizer.get_config() if self.attend_on_both and self.ba2_regularizer else None,
                  'ca2_regularizer': self.ca2_regularizer.get_config() if self.attend_on_both and self.ca2_regularizer else None,
                  'wa3_regularizer': self.wa3_regularizer.get_config() if self.attend_on_both and self.wa3_regularizer else None,
                  'Wa3_regularizer': self.Wa3_regularizer.get_config() if self.attend_on_both and self.Wa3_regularizer else None,
                  'Ua3_regularizer': self.Ua3_regularizer.get_config() if self.attend_on_both and self.Ua3_regularizer else None,
                  'ba3_regularizer': self.ba3_regularizer.get_config() if self.attend_on_both and self.ba3_regularizer else None,
                  'ca3_regularizer': self.ca3_regularizer.get_config() if self.attend_on_both and self.ca3_regularizer else None,
                  "dropout_S": self.dropout_S,
                  "dropout_T": self.dropout_T,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U,
                  "dropout_V": self.dropout_V,
                  'dropout_wa': self.dropout_wa,
                  'dropout_Wa': self.dropout_Wa,
                  'dropout_Ua': self.dropout_Ua,
                  'dropout_wa2': self.dropout_wa2 if self.attend_on_both else None,
                  'dropout_Wa2': self.dropout_Wa2 if self.attend_on_both else None,
                  'dropout_Ua2': self.dropout_Ua2 if self.attend_on_both else None,
                  'dropout_wa3': self.dropout_wa3 if self.attend_on_both else None,
                  'dropout_Wa3': self.dropout_Wa3 if self.attend_on_both else None,
                  'dropout_Ua3': self.dropout_Ua3 if self.attend_on_both else None
                  }
        base_config = super(AttLSTMCond3Inputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
