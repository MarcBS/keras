# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from .. import backend as K
from .. import activations, initializers, regularizers, constraints
from ..engine import Layer, InputSpec
from ..layers import Dense, Concatenate, TimeDistributed, Slice, Dropout


class MultiHeadAttention(Layer):
    """Multi-head attention layer. Multi-Head Attention consists of h attention layers running in parallel.

    Linearly projects queries, keys and values `h` times with different, learned
     linear projections, to the `d_k`, `d_k` and `d_v` dimensions, respectively.


    # Arguments
        n_heads: Number of attention layers that represent the linear projections.
        dmodel: model size
        mask_future: Boolean. Whether we should apply a mask to the future or not.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the attention weights computed
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Use bias in the Multi-head projections.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        3 tensors with shape: `(batch_size, input_dim)`.

    # Output shape
        The same as

    # References
        - [ Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, n_heads,
                 dmodel,
                 mask_future=False,
                 dropout=0.,
                 activation='relu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True
        assert dmodel % n_heads == 0, 'dmodel should be a multiple of the head number'
        self.n_heads = n_heads
        self.dmodel = dmodel
        self.dk = dmodel // n_heads
        self.dv = dmodel // n_heads
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.linear_v = None
        self.linear_k = None
        self.linear_q = None
        self.linear_o = None

        self.bias_v = None
        self.bias_k = None
        self.bias_q = None
        self.bias_o = None
        self.mask_future = mask_future  # If mask_future,  units that reference the future are masked.

    def build(self, input_shape):
        assert len(input_shape) == 2, 'You should pass two inputs to MultiHeadAttention: ' \
                                      'queries, keys/values.'
        query_dim = input_shape[0][2]
        key_dim = input_shape[1][2]

        self.linear_q = self.add_weight(shape=(query_dim, self.dk * self.n_heads),
                                        initializer=self.kernel_initializer,
                                        name='linear_q',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.linear_k = self.add_weight(shape=(key_dim, self.dk * self.n_heads),
                                        initializer=self.kernel_initializer,
                                        name='linear_k',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.linear_v = self.add_weight(shape=(key_dim, self.dv * self.n_heads),
                                        initializer=self.kernel_initializer,
                                        name='linear_v',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.linear_o = self.add_weight(shape=(self.dv * self.n_heads, self.dmodel),
                                        initializer=self.kernel_initializer,
                                        name='linear_o',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_q = self.add_weight(shape=(self.dk * self.n_heads,),
                                          initializer=self.bias_initializer,
                                          name='bias_q',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)

            self.bias_k = self.add_weight(shape=(self.dk * self.n_heads,),
                                          initializer=self.bias_initializer,
                                          name='bias_k',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)

            self.bias_v = self.add_weight(shape=(self.dv * self.n_heads,),
                                          initializer=self.bias_initializer,
                                          name='bias_v',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)

            self.bias_o = self.add_weight(shape=(self.dmodel,),
                                          initializer=self.bias_initializer,
                                          name='bias_o',
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint)
        else:
            self.bias_q = None
            self.bias_k = None
            self.bias_v = None
            self.bias_o = None

        if self.dropout > 0:
            self.dropout_layer = Dropout(self.dropout)

        self.built = True

    def call(self, inputs, mask=None, training=None):
        query = inputs[0]
        key = inputs[1]

        if mask is not None and mask[0] is not None:
            mask_query = K.cast(mask[0], K.dtype(query))
            query *= mask_query[:, :, None]

        if mask is not None and mask[1] is not None:
            mask_key = K.cast(mask[1], K.dtype(key))
            key *= mask_key[:, :, None]

        # Do linear projections. Shapes: batch_size, timesteps, dmodel*n_heads
        queries, keys, values = [self.activation(K.bias_add(K.dot(x, kernel), bias)) if self.use_bias else self.activation(K.dot(x, kernel))
                                 for kernel, bias, x in zip([self.linear_q, self.linear_k, self.linear_v],
                                                            [self.bias_q, self.bias_k, self.bias_v],
                                                            (query, key, key))]

        queries_ = K.concatenate([queries[:, :, i * self.dk: (i + 1) * self.dk] for i in range(self.n_heads)], axis=0)  # batch_size * n_heads, timesteps, dmodel/h
        keys_ = K.concatenate([keys[:, :, i * self.dk: (i + 1) * self.dk] for i in range(self.n_heads)], axis=0)  # batch_size * n_heads, timesteps, dmodel/h
        values_ = K.concatenate([values[:, :, i * self.dv: (i + 1) * self.dv] for i in range(self.n_heads)], axis=0)  # batch_size * n_heads, timesteps, dmodel/h

        # Scaled-Dot-Product Attention

        # Compute MatMul
        matmul = K.batch_dot(queries_, K.permute_dimensions(keys_, (0, 2, 1)), axes=[2, 1])

        # Scale it (denominator)
        scale = K.sqrt(K.cast(self.dk, K.floatx()))

        attended_heads = matmul / scale

        # Key Masking
        key_masks = K.sign(K.sum(K.abs(key), axis=-1))  # (N, T_q)
        key_masks = K.tile(key_masks, [self.n_heads, 1])  # (h*N, T_k)
        key_masks = K.tile(K.expand_dims(key_masks, 1), [1, K.shape(query)[1], 1])  # (h*N, T_q, T_k)
        paddings = K.ones_like(attended_heads) * K.variable(-2 ** 32 + 1, dtype=K.floatx())
        attended_heads = K.switch(K.equal(key_masks, 0), paddings, attended_heads)  # (h*N, T_q, T_k)

        if self.mask_future:
            diag_vals = K.ones_like(attended_heads[0, :, :])  # (T_q, T_k)
            tril = K.tril(diag_vals)  # (T_q, T_k)
            future_masks = K.tile(K.expand_dims(tril, 0), [K.shape(attended_heads)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = K.ones_like(future_masks) * K.variable(-2 ** 32 + 1, dtype=K.floatx())
            attended_heads = K.switch(K.equal(future_masks, 0), paddings, attended_heads)  # (h*N, T_q, T_k)

        # Activation (softmax)
        alphas = K.softmax_3d(attended_heads)

        if self.dropout > 0:
            alphas = self.dropout_layer(alphas)

        # Query Masking
        query_masks = K.sign(K.sum(K.abs(query), axis=-1))  # (N, T_q)
        query_masks = K.tile(query_masks, [self.n_heads, 1])  # (h*N, T_q)
        query_masks = K.tile(K.expand_dims(query_masks, -1), [1, 1, K.shape(key)[1]])  # (h*N, T_q, T_k)
        alphas = alphas * query_masks  # broadcasting. (N, T_q, C)

        # Matmul with V
        attended_heads = K.batch_dot(alphas, values_, axes=[2, 1])

        # Restore shape
        nb_samples = K.shape(attended_heads)[0] // self.n_heads
        attended_heads = K.concatenate([attended_heads[i * nb_samples: (i + 1) * nb_samples, :, :] for i in range(self.n_heads)], axis=2)  # batch_size, timesteps, dmodel

        # Apply the final linear
        output = self.activation(K.bias_add(K.dot(attended_heads, self.linear_o), self.bias_o)) if self.use_bias else self.activation(K.dot(attended_heads, self.linear_o))
        return output

    def compute_mask(self, inputs, mask=None):
        query = inputs[0]
        if mask is not None and mask[0] is not None:
            mask_query = K.cast(mask[0], K.dtype(query))
        else:
            mask_query = K.not_equal(K.sum(K.abs(query), axis=2), 0)
            mask_query = K.cast(mask_query, K.dtype(query))
        return mask_query

    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape[0]) >= 3
        assert input_shape[0][-1]
        output_shape = list(input_shape[0])
        output_shape[-1] = self.dmodel
        return tuple(output_shape)

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'dmodel': self.dmodel,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'dropout': self.dropout,
            'mask_future': self.mask_future
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# TODO: Deprecated layers. Need to update!

class Attention(Layer):
    """ Attention layer that does not depend on temporal information. The output information
        provided are the attention vectors 'alpha' over the input data.

    # Arguments
        nb_attention: number of attention mechanisms applied over the input vectors
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        dropout_Wa: float between 0 and 1.
        Wa_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        ba_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.

    # Formulation

    """

    def __init__(self, nb_attention,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh',
                 inner_activation='hard_sigmoid',
                 dropout_Wa=0.,
                 Wa_regularizer=None,
                 ba_regularizer=None,
                 **kwargs):
        self.nb_attention = nb_attention
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        # attention model learnable params
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.dropout_Wa = dropout_Wa

        if self.dropout_Wa:
            self.uses_learning_phase = True
        super(Attention, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape, ndim=3)]
        self.input_dim = input_shape[-1]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.Wa = self.init((self.input_dim, self.nb_attention),
                            name='{}_Wa'.format(self.name))

        self.ba = K.variable((np.zeros(self.nb_attention)),
                             name='{}_ba'.format(self.name))

        self.trainable_weights = [self.Wa, self.ba]

        self.regularizers = []
        # Att regularizers
        if self.Wa_regularizer:
            self.Wa_regularizer.set_param(self.Wa)
            self.regularizers.append(self.Wa_regularizer)
        if self.ba_regularizer:
            self.ba_regularizer.set_param(self.ba)
            self.regularizers.append(self.ba_regularizer)

            # if self.initial_weights is not None:
            #    self.set_weights(self.initial_weights)
            #    del self.initial_weights

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape must be:
        #   (nb_samples, temporal_or_spatial_dimensions, input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        assert len(input_shape) == 3, 'Input shape must be: (nb_samples, temporal_or_spatial_dimensions, input_dim)'

        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of temporal_or_spatial_dimensions of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        attention = self.attention_step(preprocessed_input, constants)

        return attention

    def attention_step(self, x, constants):

        # Att model dropouts
        B_Wa = constants[0]

        # AttModel (see Formulation in class header)
        # e = K.dot(K.tanh(K.dot(x * B_W, self.W) + self.b) * B_w, self.w)

        # Attention spatial weights 'alpha'
        # e = K.permute_dimensions(e, (0,2,1))
        # alpha = K.softmax_3d(e)
        # alpha = K.permute_dimensions(alpha, (0,2,1))

        # Attention class weights 'beta'
        # beta = K.sigmoid(K.dot(alpha * B_Wa, self.Wa) + self.ba)
        beta = K.sigmoid(K.dot(x * B_Wa, self.Wa) + self.ba)

        # TODO: complete formulas in class description
        return beta

    def get_constants(self, x):
        constants = []

        # AttModel

        if 0 < self.dropout_Wa < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_Wa = K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        return constants

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[:2]) + [self.nb_attention])

    def get_config(self):
        config = {'nb_attention': self.nb_attention,
                  'kernel_initializer': self.init.__name__,
                  'recurrent_initializer': self.inner_init.__name__,
                  'forget_bias_initializer': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'recurrent_activation': self.inner_activation.__name__,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'dropout_Wa': self.dropout_Wa}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftAttention(Layer):
    """ Simple soft Attention layer
    The output information provided are the attended input an the attention weights 'alpha' over the input data.

    # Arguments
        att_dim: Soft alignment MLP dimension
        sum_weighted_output: Boolean, whether to sum the weigthed output
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        activation: ctivation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        dropout_Wa: float between 0 and 1.
        dropout_Ua: float between 0 and 1.
        wa_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        Wa_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        Ua_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        ba_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        ca_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.

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

    """

    def __init__(self,
                 att_dim,
                 sum_weighted_output=True,
                 init='glorot_uniform',
                 activation='tanh',
                 dropout_Wa=0.,
                 dropout_Ua=0.,
                 wa_regularizer=None,
                 Wa_regularizer=None,
                 Ua_regularizer=None,
                 ba_regularizer=None,
                 ca_regularizer=None,
                 **kwargs):
        self.att_dim = att_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.sum_weighted_output = sum_weighted_output

        self.dropout_Wa, self.dropout_Ua = dropout_Wa, dropout_Ua

        # attention model learnable params
        self.wa_regularizer = regularizers.get(wa_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.Ua_regularizer = regularizers.get(Ua_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.ca_regularizer = regularizers.get(ca_regularizer)

        if self.dropout_Wa or self.dropout_Ua:
            self.uses_learning_phase = True
        super(SoftAttention, self).__init__(**kwargs)
        # self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        assert len(input_shape) == 2, 'You should pass two inputs to SoftAttention '
        self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
        self.input_steps = input_shape[0][1]
        self.input_dim = input_shape[0][2]
        self.context_dim = input_shape[1][1]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.wa = self.add_weight((self.att_dim,),
                                  initializer=self.init,
                                  name='{}_wa'.format(self.name),
                                  regularizer=self.wa_regularizer)

        self.Wa = self.add_weight((self.input_dim, self.att_dim),
                                  initializer=self.init,
                                  name='{}_Wa'.format(self.name),
                                  regularizer=self.Wa_regularizer)

        self.Ua = self.add_weight((self.context_dim, self.att_dim),
                                  initializer=self.init,
                                  name='{}_Ua'.format(self.name),
                                  regularizer=self.Ua_regularizer)

        self.ba = self.add_weight(self.att_dim,
                                  initializer='zero',
                                  name='{}_ba'.format(self.name),
                                  regularizer=self.ba_regularizer)

        self.ca = self.add_weight(self.input_steps,
                                  initializer='zero',
                                  name='{}_ca'.format(self.name),
                                  regularizer=self.ca_regularizer)

        self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, self.ca]  # AttModel parameters

        self.built = True

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape must be:
        #   (nb_samples, temporal_or_spatial_dimensions, input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context = x[1]
        assert len(input_shape) == 3, 'Input shape must be: (nb_samples, temporal_or_spatial_dimensions, input_dim)'

        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of temporal_or_spatial_dimensions of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        constants = self.get_constants(state_below, mask[1])
        preprocessed_input = self.preprocess_input(state_below)

        [attended_representation, alphas] = self.attention_step(preprocessed_input, constants)

        return [attended_representation, alphas]

    def attention_step(self, x, constants):
        # Att model dropouts
        B_Wa = constants[0]  # Dropout Wa

        pctx_ = constants[1]  # Original context

        # Attention model (see Formulation in class header)
        p_state_ = K.dot(x * B_Wa[0], self.Wa)
        pctx_ = self.activation(pctx_[:, None, :] + p_state_)
        e = K.dot(pctx_, self.wa) + self.ca
        alphas_shape = e.shape
        alphas = K.softmax(e.reshape([alphas_shape[0], alphas_shape[1]]))

        # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        ctx_ = x * alphas[:, :, None]
        if self.sum_weighted_output:
            ctx_ = (ctx_).sum(axis=1)
        return [ctx_, alphas]

    def get_constants(self, x, mask_context):
        constants = []

        # constants[0]
        if 0 < self.dropout_Wa < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_Wa = [K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)]
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        # constants[1]
        if 0 < self.dropout_Ua < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, self.context.shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.dropout_Ua), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.Ua) + self.ba
        else:
            pctx = K.dot(self.context, self.Ua) + self.ba
        constants.append(pctx)

        return constants

    def get_output_shape_for(self, input_shape):
        if self.sum_weighted_output:
            dim_x_att = (input_shape[0][0], input_shape[0][2])
        else:
            dim_x_att = (input_shape[0])
        dim_alpha_att = (input_shape[0][0], input_shape[0][1])
        main_out = [dim_x_att, dim_alpha_att]
        return main_out

    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def get_config(self):
        config = {'att_units': self.att_dim,
                  'kernel_initializer': self.init.__name__,
                  'activation': self.activation.__name__,
                  'sum_weighted_output': self.sum_weighted_output,
                  'wa_regularizer': self.wa_regularizer.get_config() if self.wa_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'Ua_regularizer': self.Ua_regularizer.get_config() if self.Ua_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'ca_regularizer': self.ca_regularizer.get_config() if self.ca_regularizer else None,
                  'dropout_Wa': self.dropout_Wa,
                  'dropout_Ua': self.dropout_Ua,
                  }
        base_config = super(SoftAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftMultistepsAttention(Layer):
    """ Multi timesteps soft Attention layer
    The output information provided are the attended input an the attention weights 'alpha' over the input data.

    # Arguments
        att_dim: Soft alignment MLP dimension
        sum_weighted_output: Boolean, whether to sum the weigthed output
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        activation: ctivation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        return_sequences: whether to return sequences or not
        dropout_Wa: float between 0 and 1.
        dropout_Ua: float between 0 and 1.
        wa_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        Wa_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        Ua_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        ba_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        ca_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.

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

    """

    def __init__(self, att_dim, sum_weighted_output=True,
                 init='glorot_uniform', activation='tanh',
                 return_sequences=True,
                 dropout_Wa=0.,
                 dropout_Ua=0.,
                 wa_regularizer=None,
                 Wa_regularizer=None,
                 Ua_regularizer=None,
                 ba_regularizer=None,
                 ca_regularizer=None,
                 **kwargs):
        self.att_dim = att_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.sum_weighted_output = sum_weighted_output
        self.return_sequences = return_sequences

        self.dropout_Wa, self.dropout_Ua = dropout_Wa, dropout_Ua

        # attention model learnable params
        self.wa_regularizer = regularizers.get(wa_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.Ua_regularizer = regularizers.get(Ua_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.ca_regularizer = regularizers.get(ca_regularizer)

        if self.dropout_Wa or self.dropout_Ua:
            self.uses_learning_phase = True
        super(SoftMultistepsAttention, self).__init__(**kwargs)
        # self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        assert len(input_shape) == 2, 'You should pass two inputs to SoftMultistepsAttention '
        self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
        self.input_steps = input_shape[0][1]
        self.input_dim = input_shape[0][2]
        self.context_dim = input_shape[1][2]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.wa = self.add_weight((self.att_dim,),
                                  initializer=self.init,
                                  name='{}_wa'.format(self.name),
                                  regularizer=self.wa_regularizer)

        self.Wa = self.add_weight((self.input_dim, self.att_dim),
                                  initializer=self.init,
                                  name='{}_Wa'.format(self.name),
                                  regularizer=self.Wa_regularizer)

        self.Ua = self.add_weight((self.context_dim, self.att_dim),
                                  initializer=self.init,
                                  name='{}_Ua'.format(self.name),
                                  regularizer=self.Ua_regularizer)

        self.ba = self.add_weight(self.att_dim,
                                  initializer='zero',
                                  name='{}_ba'.format(self.name),
                                  regularizer=self.ba_regularizer)

        self.ca = self.add_weight(self.input_steps,
                                  initializer='zero',
                                  name='{}_ca'.format(self.name),
                                  regularizer=self.ca_regularizer)

        self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, self.ca]  # AttModel parameters

        self.built = True

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape must be:
        #   (nb_samples, temporal_or_spatial_dimensions, input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context = x[1]
        assert len(input_shape) == 3, 'Input shape must be: (nb_samples, temporal_or_spatial_dimensions, input_dim)'

        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of temporal_or_spatial_dimensions of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        constants = self.get_constants(state_below, mask[1])
        preprocessed_input = self.preprocess_input(state_below)

        last_output, outputs, states = K.rnn(self.attention_step, preprocessed_input,
                                             [None, None],  # self.get_extra_states(x),
                                             go_backwards=False,
                                             mask=None,
                                             # mask[1], #TODO: What does this mask mean? How should it be applied?
                                             constants=constants,
                                             unroll=False,
                                             input_length=self.input_steps)

        if self.return_sequences:
            return states
        return [states[0][-1], states[1][-1]]

    def get_initial_states(self, x):

        pctx_state = K.zeros_like(x[1])  # (samples, height*width, features_in)
        pctx_state = K.sum(pctx_state, axis=(-1))
        alpha_state = pctx_state
        pctx_state = K.expand_dims(pctx_state, dim=-1)
        pctx_state = K.repeat_elements(pctx_state, self.att_dim, -1)

        return [pctx_state, alpha_state]

    def attention_step(self, x, constants):
        # Att model dropouts
        B_Wa = constants[0]  # Dropout Wa

        pctx_ = constants[1]  # Original context

        # Attention model (see Formulation in class header)
        p_state_ = K.dot(x * B_Wa[0], self.Wa)
        pctx_ = self.activation(pctx_ + p_state_[:, None, :])
        e = K.dot(pctx_, self.wa) + self.ca
        alphas_shape = e.shape
        alphas = K.softmax(e.reshape([alphas_shape[0], alphas_shape[1]]))

        # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        ctx_ = x * alphas[:, :, None]
        if self.sum_weighted_output:
            ctx_ = (ctx_).sum(axis=1)
        return ctx_, [ctx_, alphas]

    def get_constants(self, x, mask_context):
        constants = []

        # constants[0]
        if 0 < self.dropout_Wa < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_Wa = [K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)]
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        # constants[1]
        if 0 < self.dropout_Ua < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, self.context.shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.dropout_Ua), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.Ua) + self.ba
        else:
            pctx = K.dot(self.context, self.Ua) + self.ba
        constants.append(pctx)

        return constants

    def get_output_shape_for(self, input_shape):
        if self.sum_weighted_output:
            dim_x_att = (input_shape[1][0], input_shape[0][1], self.att_dim)
        else:
            dim_x_att = (input_shape[1][0], input_shape[0][1], input_shape[1][1], self.att_dim)
        dim_alpha_att = (input_shape[1][0], input_shape[0][1], input_shape[1][1])
        main_out = [dim_x_att, dim_alpha_att]
        return main_out

    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def get_config(self):
        config = {'att_units': self.att_dim,
                  'kernel_initializer': self.init.__name__,
                  'activation': self.activation.__name__,
                  'sum_weighted_output': self.sum_weighted_output,
                  'return_sequences': self.return_sequences,
                  'wa_regularizer': self.wa_regularizer.get_config() if self.wa_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'Ua_regularizer': self.Ua_regularizer.get_config() if self.Ua_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'ca_regularizer': self.ca_regularizer.get_config() if self.ca_regularizer else None,
                  'dropout_Wa': self.dropout_Wa,
                  'dropout_Ua': self.dropout_Ua,
                  }
        base_config = super(SoftMultistepsAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionComplex(Layer):
    """ Attention layer that does not depend on temporal information. The output information
        provided are the attention vectors 'alpha' over the input data.

    # Arguments
        nb_attention: number of attention mechanisms applied over the input vectors
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        dropout_w: float between 0 and 1.
        dropout_W: float between 0 and 1.
        dropout_Wa: float between 0 and 1.
        w_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        Wa_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        ba_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.

    # Formulation

    """

    def __init__(self, nb_attention,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh',
                 inner_activation='hard_sigmoid',
                 dropout_w=0.,
                 dropout_W=0.,
                 dropout_Wa=0.,
                 w_regularizer=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 Wa_regularizer=None,
                 ba_regularizer=None,
                 **kwargs):
        self.nb_attention = nb_attention
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        # attention model learnable params
        self.w_regularizer = regularizers.get(w_regularizer)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.dropout_w, self.dropout_W, self.dropout_Wa = dropout_w, dropout_W, dropout_Wa

        if self.dropout_w or self.dropout_W or self.dropout_Wa:
            self.uses_learning_phase = True
        super(AttentionComplex, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape, ndim=3)]
        self.input_dim = input_shape[-1]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        # self.w = self.add_weight((self.input_dim,),
        self.w = self.add_weight((self.input_dim, self.nb_attention),
                                 initializer=self.init,
                                 name='{}_w'.format(self.name),
                                 regularizer=self.w_regularizer)

        # self.W = self.add_weight((self.input_dim, self.nb_attention, self.input_dim),
        self.W = self.add_weight((self.input_dim, self.input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)

        self.b = self.add_weight(self.input_dim,
                                 initializer='zero',
                                 regularizer=self.b_regularizer)

        """
        self.Wa = self.add_weight((self.nb_attention, self.nb_attention),
                                 initializer=self.kernel_initializer,
                                 name='{}_Wa'.format(self.name),
                                 regularizer=self.Wa_regularizer)

        self.ba = self.add_weight(self.input_dim,
                                  initializer= 'zero',
                                  regularizer=self.ba_regularizer)

        self.trainable_weights = [self.w, self.W, self.b, self.Wa, self.ba] # AttModel parameters
        """
        self.trainable_weights = [self.w, self.W, self.b]
        # if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape must be:
        #   (nb_samples, temporal_or_spatial_dimensions, input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        assert len(input_shape) == 3, 'Input shape must be: (nb_samples, temporal_or_spatial_dimensions, input_dim)'

        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of temporal_or_spatial_dimensions of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        attention = self.attention_step(preprocessed_input, constants)

        return attention

    def attention_step(self, x, constants):

        # Att model dropouts
        B_w = constants[0]
        B_W = constants[1]
        B_Wa = constants[2]

        # AttModel (see Formulation in class header)
        e = K.dot(K.tanh(K.dot(x * B_W, self.W) + self.b) * B_w, self.w)
        return e

        # Attention spatial weights 'alpha'
        # e = e.dimshuffle((0, 2, 1))
        e = K.permute_dimensions(e, (0, 2, 1))
        # alpha = K.softmax(e)
        # return alpha
        alpha = K.softmax_3d(e)
        alpha = K.permute_dimensions(alpha, (0, 2, 1))

        return alpha

        # alpha = alpha.dimshuffle((0,2,1))

        # Attention class weights 'beta'
        beta = K.sigmoid(K.dot(alpha * B_Wa, self.Wa) + self.ba)
        # beta = K.softmax_3d(K.dot(alpha * B_Wa, self.Wa) + self.ba)

        # Sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        # x_att = (x * alpha[:,:,None]).sum(axis=1)

        # TODO: complete formulas in class description

        return beta

    def get_constants(self, x):
        constants = []

        # AttModel
        if 0 < self.dropout_w < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_w = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
            constants.append(B_w)
        else:
            constants.append(K.cast_to_floatx(1.))

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))

        if 0 < self.dropout_Wa < 1:
            input_shape = self.input_spec[0].shape
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * self.nb_attention, 2)
            B_Wa = K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)
            constants.append(B_Wa)
        else:
            constants.append(K.cast_to_floatx(1.))

        return constants

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[:2]) + [self.nb_attention])

    def get_config(self):
        config = {'nb_attention': self.nb_attention,
                  'kernel_initializer': self.init.__name__,
                  'recurrent_initializer': self.inner_init.__name__,
                  'forget_bias_initializer': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'recurrent_activation': self.inner_activation.__name__,
                  'w_regularizer': self.w_regularizer.get_config() if self.w_regularizer else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'dropout_w': self.dropout_w,
                  'dropout_W': self.dropout_W,
                  'dropout_Wa': self.dropout_Wa}
        base_config = super(AttentionComplex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvAtt(Layer):
    """Convolution operator for filtering windows of two-dimensional inputs with Attention mechanism.
    The first input corresponds to the image and the second input to the weighting vector (which contains a set of steps).
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures. An additional input for modulating the attention is required.

    # Examples

    ```python
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```
    # Arguments
            nb_embedding: Number of convolution filters to use.
            nb_glimpses: Number of glimpses to take
            concat_timesteps: Boolean. Whether we concatenate timesteps or not.
            init: name of initialization function for the weights of the layer
                (see [initializations](../initializations.md)), or alternatively,
                Theano function to use for weights initialization.
                This parameter is only relevant if you don't pass
                a `weights` argument.
            activation: name of activation function to use
                (see [activations](../activations.md)),
                or alternatively, elementwise Theano function.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: a(x) = x).
            weights: list of numpy arrays to set as initial weights.
            return_states: boolean. Whether we return states or not
            border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)
            dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
                (the depth) is at index 1, in 'tf' mode is it at index 3.
                It defaults to the `image_dim_ordering` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "tf".
            W_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the main weights matrix.
            U_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the main weights matrix.
            V_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the main weights matrix.
            b_regularizer: instance of [WeightRegularizer](../regularizers.md),
                applied to the bias.
            activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
                applied to the network output.
            W_constraint: instance of the [constraints](../constraints.md) module
                (eg. maxnorm, nonneg), applied to the main weights matrix.
            U_constraint: instance of the [constraints](../constraints.md) module
                (eg. maxnorm, nonneg), applied to the main weights matrix.
            V_constraint: instance of the [constraints](../constraints.md) module
                (eg. maxnorm, nonneg), applied to the main weights matrix.
            b_constraint: instance of the [constraints](../constraints.md) module,
                applied to the bias.
            W_learning_rate_multiplier: multiplier of the learning rate for W
            b_learning_rate_multiplier: multiplier of the learning rate for W
            bias: whether to include a bias
                (i.e. make the layer affine rather than linear).

        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if dim_ordering='tf'.
            and 4D tensor with shape:
            `(samples, steps, features)`

        # Output shape
            4D tensor with shape:
            `(samples, nb_filter, rows, cols)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, rows, cols, nb_filter)` if dim_ordering='tf'.
            `rows` and `cols` values might have changed due to padding.
        """

    def __init__(self, nb_embedding, nb_glimpses=1, concat_timesteps=True,
                 init='glorot_uniform', activation=None, weights=None, return_states=True,
                 border_mode='valid', dim_ordering='default',
                 W_regularizer=None, U_regularizer=None, V_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, U_constraint=None, V_constraint=None, b_constraint=None,
                 W_learning_rate_multiplier=None, b_learning_rate_multiplier=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for Convolution2D:', border_mode)
        self.nb_embedding = nb_embedding
        self.nb_glimpses = nb_glimpses
        self.concat_timesteps = concat_timesteps  # if True output_size=(samples, nb_glimpses*num_timesteps, rows, cols)
        # if False output_size=(samples, num_timesteps, nb_glimpses, rows, cols)
        self.nb_row = 1
        self.nb_col = 1
        self.return_states = return_states
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple((1, 1))
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.nb_glimpses > 0:
            self.U_regularizer = regularizers.get(U_regularizer)
        else:
            self.U_regularizer = None
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        if self.nb_glimpses > 0:
            self.U_constraint = constraints.get(U_constraint)
        else:
            self.U_constraint = None
        self.V_constraint = constraints.get(V_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier, self.b_learning_rate_multiplier]

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        self.supports_masking = True
        super(ConvAtt, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_words = input_shape[1][1]
        if self.dim_ordering == 'th':
            img_size = input_shape[0][1]
            qst_size = input_shape[1][2]
            if self.nb_glimpses > 0:
                self.U_shape = (self.nb_glimpses, self.nb_embedding, self.nb_row, self.nb_col)
            self.V_shape = (qst_size, self.nb_embedding)
            self.W_shape = (self.nb_embedding, img_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            img_size = input_shape[0][3]
            qst_size = input_shape[1][2]
            if self.nb_glimpses > 0:
                self.U_shape = (self.nb_row, self.nb_col, self.nb_embedding, self.nb_glimpses)
            self.V_shape = (qst_size, self.nb_embedding)
            self.W_shape = (self.nb_row, self.nb_col, img_size, self.nb_embedding)
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        if self.nb_glimpses > 0:
            self.U = self.add_weight(self.U_shape,
                                     initializer=self.init,
                                     name='{}_U'.format(self.name),
                                     regularizer=self.U_regularizer,
                                     constraint=self.U_constraint)
        else:
            self.U = None
        self.V = self.add_weight(self.V_shape,
                                 initializer=self.init,
                                 name='{}_V'.format(self.name),
                                 regularizer=self.V_regularizer,
                                 constraint=self.V_constraint)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.nb_embedding,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x):
        return K.dot(x, self.V)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[0][2]
            cols = input_shape[0][3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[0][1]
            cols = input_shape[0][2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        """
        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])
        """

        # return (input_shape[0][0], self.num_words, self.nb_embedding, rows, cols)

        if self.return_states:
            if False:  # self.nb_glimpses > 0:
                if self.concat_timesteps:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.nb_glimpses * self.num_words, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], rows, cols, self.nb_glimpses * self.num_words)
                else:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.num_words, self.nb_glimpses, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], self.num_words, rows, cols, self.nb_glimpses)
            else:
                if self.concat_timesteps:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.nb_embedding * self.num_words, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], rows, cols, self.nb_embedding * self.num_words)
                else:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.num_words, self.nb_embedding, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], self.num_words, rows, cols, self.nb_embedding)

        else:
            if False:  # self.nb_glimpses > 0:
                if self.dim_ordering == 'th':
                    return (input_shape[0][0], self.nb_glimpses, rows, cols)
                elif self.dim_ordering == 'tf':
                    return (input_shape[0][0], rows, cols, self.nb_glimpses)
            else:
                if self.dim_ordering == 'th':
                    return (input_shape[0][0], self.nb_embedding, rows, cols)
                elif self.dim_ordering == 'tf':
                    return (input_shape[0][0], rows, cols, self.nb_embedding)

    def call(self, x, mask=None):

        preprocessed_img = K.conv2d(x[0], self.W, strides=self.subsample,
                                    border_mode=self.border_mode,
                                    dim_ordering=self.dim_ordering,
                                    filter_shape=self.W_shape)

        preprocessed_input = self.preprocess_input(x[1])  # TODO: Dropout?

        if self.bias:
            if self.dim_ordering == 'th':
                preprocessed_img += K.reshape(self.b, (1, self.nb_embedding, 1, 1))
            elif self.dim_ordering == 'tf':
                preprocessed_img += K.reshape(self.b, (1, 1, 1, self.nb_embedding))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             self.get_initial_states(x),
                                             go_backwards=False,
                                             mask=None,
                                             # mask[1], #TODO: What does this mask mean? How should it be applied?
                                             constants=[preprocessed_img],
                                             unroll=False,
                                             input_length=self.num_words)

        if self.return_states:
            # Join temporal and glimpses dimensions
            if self.concat_timesteps:
                outputs = K.permute_dimensions(outputs, (0, 3, 4, 2, 1))
                shp = outputs.shape
                outputs = K.reshape(outputs, (shp[0], shp[1], shp[2], -1))
                outputs = K.permute_dimensions(outputs, (0, 3, 1, 2))

            return outputs

        else:
            return last_output

    def get_initial_states(self, x):

        initial_state = K.zeros_like(x[0])  # (samples, features_in, height, width)
        initial_state = K.sum(initial_state, axis=(1))
        initial_state = K.expand_dims(initial_state, dim=1)
        """
        if self.nb_glimpses > 0:
            initial_state = K.repeat_elements(initial_state, self.nb_glimpses, 1)
        else:
            initial_state = K.repeat_elements(initial_state, self.nb_embedding, 1)
        """
        initial_state = K.repeat_elements(initial_state, self.nb_embedding, 1)

        return [initial_state]
        # return [initial_state, initial_state] # (samples, nb_glimpses, height, width)

    def step(self, x, states):
        context = states[1]

        activation_t = K.tanh(context + x[:, :, None, None])

        if self.nb_glimpses > 0:
            e_t = K.conv2d(activation_t,
                           self.U,
                           strides=(1, 1),
                           border_mode='valid',
                           dim_ordering=self.dim_ordering,
                           filter_shape=self.U_shape)
        else:
            e_t = activation_t

        # Apply softmax on att. weights
        e_t_reshaped = e_t.sum(axis=1)
        alphas_shape = e_t_reshaped.shape
        e_t_reshaped = e_t_reshaped.reshape([alphas_shape[0], alphas_shape[1] * alphas_shape[2]])
        alphas = K.softmax(e_t_reshaped)
        alphas = alphas.reshape([alphas_shape[0], alphas_shape[1], alphas_shape[2]])

        # Weight input image vectors according to alphas
        attended_ctx = context * alphas[:, None, :, :]

        ############################################################

        """
        alphas_shape = e_t.shape
        e_t_reshaped = e_t.reshape([alphas_shape[0], alphas_shape[1], alphas_shape[2]*alphas_shape[3]])
        e_t_reshaped = K.permute_dimensions(e_t_reshaped, [0,2,1])
        alphas = K.softmax_3d(e_t_reshaped)
        alphas = K.permute_dimensions(alphas, [0, 2, 1])
        alphas = alphas.reshape([alphas_shape[0], alphas_shape[1], alphas_shape[2], alphas_shape[3]])

        # Weight input image vectors according to alphas
        attended_ctx = context * alphas
        #if self.sum_weighted_output:
        #    attended_ctx = (attended_ctx).sum(axis=1)
        """

        # return e_t, [e_t]
        return attended_ctx, [attended_ctx]  # [attended_ctx, e_t]

    def compute_mask(self, input, mask):
        if self.nb_glimpses > 0:
            out_mask = K.repeat(mask[1], self.nb_glimpses)
        else:
            out_mask = K.repeat(mask[1], self.nb_embedding)

        out_mask = K.repeat(mask[1], self.nb_embedding)

        out_mask = K.flatten(out_mask)
        return out_mask

    def get_config(self):
        config = {'nb_embedding': self.nb_embedding,
                  'nb_glimpses': self.nb_glimpses,
                  'concat_timesteps': self.concat_timesteps,
                  'return_state': self.return_states,
                  'kernel_initializer': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'U_constraint': self.U_constraint.get_config() if self.U_constraint else None,
                  'V_constraint': self.V_constraint.get_config() if self.V_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'b_learning_rate_multiplier': self.b_learning_rate_multiplier,
                  'bias': self.bias}
        base_config = super(ConvAtt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def set_lr_multipliers(self, W_learning_rate_multiplier, b_learning_rate_multiplier):
        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,
                                          self.b_learning_rate_multiplier]


class ConvCoAtt(Layer):
    """Convolution operator for filtering windows of two-dimensional inputs with Attention mechanism.
    The first input corresponds to the image and the second input to the weighting vector (which contains a set of steps).
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures. An additional input for modulating the attention is required.

    # Examples

    ```python
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```
    # Arguments
            nb_embedding: Number of convolution filters to use.
            nb_glimpses: Number of glimpses to take
            concat_timesteps: Boolean. Whether we concatenate timesteps or not.
            init: name of initialization function for the weights of the layer
                (see [initializations](../initializations.md)), or alternatively,
                Theano function to use for weights initialization.
                This parameter is only relevant if you don't pass
                a `weights` argument.
            activation: name of activation function to use
                (see [activations](../activations.md)),
                or alternatively, elementwise Theano function.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: a(x) = x).
            weights: list of numpy arrays to set as initial weights.
            return_states: boolean. Whether we return states or not
            border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)
            dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
                (the depth) is at index 1, in 'tf' mode is it at index 3.
                It defaults to the `image_dim_ordering` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "tf".
            W_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the main weights matrix.
            U_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the main weights matrix.
            V_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the main weights matrix.
            b_regularizer: instance of [WeightRegularizer](../regularizers.md),
                applied to the bias.
            activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
                applied to the network output.
            W_constraint: instance of the [constraints](../constraints.md) module
                (eg. maxnorm, nonneg), applied to the main weights matrix.
            U_constraint: instance of the [constraints](../constraints.md) module
                (eg. maxnorm, nonneg), applied to the main weights matrix.
            V_constraint: instance of the [constraints](../constraints.md) module
                (eg. maxnorm, nonneg), applied to the main weights matrix.
            b_constraint: instance of the [constraints](../constraints.md) module,
                applied to the bias.
            W_learning_rate_multiplier: multiplier of the learning rate for W
            b_learning_rate_multiplier: multiplier of the learning rate for W
            bias: whether to include a bias
                (i.e. make the layer affine rather than linear).

        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if dim_ordering='tf'.
            and 4D tensor with shape:
            `(samples, steps, features)`

        # Output shape
            4D tensor with shape:
            `(samples, nb_filter, rows, cols)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, rows, cols, nb_filter)` if dim_ordering='tf'.
            `rows` and `cols` values might have changed due to padding.
        """

    def __init__(self, nb_embedding, nb_glimpses=1, concat_timesteps=True,
                 init='glorot_uniform', activation=None, weights=None, return_states=True,
                 border_mode='valid', dim_ordering='default',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, U_constraint=None, b_constraint=None,
                 W_learning_rate_multiplier=None, b_learning_rate_multiplier=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for Convolution2D:', border_mode)
        self.nb_embedding = nb_embedding
        self.nb_glimpses = nb_glimpses

        self.return_states = return_states  # if True see self.concat_timesteps
        # if False output_size=(samples, nb_glimpses, rows, cols)

        self.concat_timesteps = concat_timesteps  # if True output_size=(samples, nb_glimpses*num_timesteps, rows, cols)
        # if False output_size=(samples, num_timesteps, nb_glimpses, rows, cols)
        self.nb_row = 1
        self.nb_col = 1
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple((1, 1))
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        if self.nb_glimpses > 0:
            self.U_regularizer = regularizers.get(U_regularizer)
        else:
            self.U_regularizer = None
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        if self.nb_glimpses > 0:
            self.U_constraint = constraints.get(U_constraint)
        else:
            self.U_constraint = None
        self.b_constraint = constraints.get(b_constraint)

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier, self.b_learning_rate_multiplier]

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        self.supports_masking = True
        super(ConvCoAtt, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_words = input_shape[1][1]
        if self.dim_ordering == 'th':
            img_size = input_shape[0][1]
            qst_size = input_shape[1][2]
            self.num_row = input_shape[0][2]
            self.num_col = input_shape[0][3]
            if self.nb_glimpses > 0:
                self.U_shape = (self.nb_glimpses, self.nb_embedding, self.nb_row, self.nb_col)
            self.W_shape = (self.nb_embedding, img_size + qst_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            img_size = input_shape[0][3]
            qst_size = input_shape[1][2]
            self.num_row = input_shape[0][1]
            self.num_col = input_shape[0][2]
            if self.nb_glimpses > 0:
                self.U_shape = (self.nb_row, self.nb_col, self.nb_embedding, self.nb_glimpses)
            self.W_shape = (self.nb_row, self.nb_col, img_size + qst_size, self.nb_embedding)
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        if self.nb_glimpses > 0:
            self.U = self.add_weight(self.U_shape,
                                     initializer=self.init,
                                     name='{}_U'.format(self.name),
                                     regularizer=self.U_regularizer,
                                     constraint=self.U_constraint)
        else:
            self.U = None
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.nb_embedding,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x):
        return x

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[0][2]
            cols = input_shape[0][3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[0][1]
            cols = input_shape[0][2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        """
        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])
        """

        # return (input_shape[0][0], self.num_words, self.nb_embedding, rows, cols)

        if self.return_states:
            if self.nb_glimpses > 0:
                if self.concat_timesteps:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.nb_glimpses * self.num_words, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], rows, cols, self.nb_glimpses * self.num_words)
                else:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.num_words, self.nb_glimpses, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], self.num_words, rows, cols, self.nb_glimpses)
            else:
                if self.concat_timesteps:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.nb_embedding * self.num_words, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], rows, cols, self.nb_embedding * self.num_words)
                else:
                    if self.dim_ordering == 'th':
                        return (input_shape[0][0], self.num_words, self.nb_embedding, rows, cols)
                    elif self.dim_ordering == 'tf':
                        return (input_shape[0][0], self.num_words, rows, cols, self.nb_embedding)

        else:
            if self.nb_glimpses > 0:
                if self.dim_ordering == 'th':
                    return (input_shape[0][0], self.nb_glimpses, rows, cols)
                elif self.dim_ordering == 'tf':
                    return (input_shape[0][0], rows, cols, self.nb_glimpses)
            else:
                if self.dim_ordering == 'th':
                    return (input_shape[0][0], self.nb_embedding, rows, cols)
                elif self.dim_ordering == 'tf':
                    return (input_shape[0][0], rows, cols, self.nb_embedding)

    def call(self, x, mask=None):

        preprocessed_img = x[0]

        preprocessed_input = self.preprocess_input(x[1])

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             self.get_initial_states(x),
                                             go_backwards=False,
                                             mask=None,
                                             # mask[1], #TODO: What does this mask mean? How should it be applied?
                                             constants=[preprocessed_img],
                                             unroll=False,
                                             input_length=self.num_words)

        if self.return_states:
            # Join temporal and glimpses dimensions
            if self.concat_timesteps:
                outputs = K.permute_dimensions(outputs, (0, 3, 4, 2, 1))
                shp = outputs.shape
                outputs = K.reshape(outputs, (shp[0], shp[1], shp[2], -1))
                outputs = K.permute_dimensions(outputs, (0, 3, 1, 2))

            return outputs

        else:
            return last_output

    def get_initial_states(self, x):

        initial_state = K.zeros_like(x[0])  # (samples, features_in, height, width)
        initial_state = K.sum(initial_state, axis=(1))
        initial_state = K.expand_dims(initial_state, dim=1)
        initial_state = K.repeat_elements(initial_state, self.nb_embedding, 1)

        return [initial_state]
        # return [initial_state, initial_state] # (samples, nb_glimpses, height, width)

    def step(self, x, states):
        context = states[1]

        if self.dim_ordering == 'th':
            x = K.repeatRdim(x, self.num_row, axis=2)
            x = K.repeatRdim(x, self.num_col, axis=3)
            concat_axis = 1
        elif self.dim_ordering == 'tf':
            x = K.repeatRdim(x, self.num_row, axis=1)
            x = K.repeatRdim(x, self.num_col, axis=2)
            concat_axis = 3
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        word_ctx = K.concatenate([x, context], axis=concat_axis)
        word_ctx = K.conv2d(word_ctx,
                            self.W,
                            strides=(1, 1),
                            border_mode='valid',
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)

        if self.bias:
            if self.dim_ordering == 'th':
                word_ctx = word_ctx + K.reshape(self.b, (1, self.nb_embedding, 1, 1))
            elif self.dim_ordering == 'tf':
                word_ctx = word_ctx + K.reshape(self.b, (1, 1, 1, self.nb_embedding))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        activation_t = K.relu(word_ctx)

        if self.nb_glimpses > 0:
            e_t = K.conv2d(activation_t,
                           self.U,
                           strides=(1, 1),
                           border_mode='valid',
                           dim_ordering=self.dim_ordering,
                           filter_shape=self.U_shape)
        else:
            e_t = activation_t

        # Apply softmax on att. weights
        e_t_reshaped = e_t.sum(axis=1)
        alphas_shape = e_t_reshaped.shape
        e_t_reshaped = e_t_reshaped.reshape([alphas_shape[0], alphas_shape[1] * alphas_shape[2]])
        alphas = K.softmax(e_t_reshaped)
        alphas = alphas.reshape([alphas_shape[0], alphas_shape[1], alphas_shape[2]])

        # Weight input image vectors according to alphas
        # attended = context * alphas[:, None, :, :]
        attended = word_ctx * alphas[:, None, :, :]

        # return e_t, [e_t]
        return attended, [attended]  # [attended, e_t]

    def compute_mask(self, input, mask):
        if self.nb_glimpses > 0:
            out_mask = K.repeat(mask[1], self.nb_glimpses)
        else:
            out_mask = K.repeat(mask[1], self.nb_embedding)

        out_mask = K.repeat(mask[1], self.nb_embedding)

        out_mask = K.flatten(out_mask)
        return out_mask

    def get_config(self):
        config = {'nb_embedding': self.nb_embedding,
                  'nb_glimpses': self.nb_glimpses,
                  'concat_timesteps': self.concat_timesteps,
                  'return_state': self.return_states,
                  'kernel_initializer': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'U_constraint': self.U_constraint.get_config() if self.U_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'b_learning_rate_multiplier': self.b_learning_rate_multiplier,
                  'bias': self.bias}
        base_config = super(ConvCoAtt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def set_lr_multipliers(self, W_learning_rate_multiplier, b_learning_rate_multiplier):
        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,
                                          self.b_learning_rate_multiplier]
