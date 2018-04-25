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
               e_i(t) = h_tm1' · Ua * x_i

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
    elif hasattr(attention_mode, '__call__'):
        e = attention_mode(h_tm1, pctx_, context, att_dp_mask, attention_recurrent_kernel,
                           attention_context_wa, bias_ca, mask_context)
    else:
        raise NotImplementedError('The attention mode ' + attention_mode + ' is not implemented.')

    if K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
        e = K.cast(mask_context, K.dtype(e)) * e
    alphas = K.softmax(K.reshape(e, [K.shape(e)[0], K.shape(e)[1]]))

    # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
    ctx_ = K.sum(context * alphas[:, :, None], axis=1)

    return ctx_, alphas


class StackedRNNCells(Layer):
    """Wrapper allowing a stack of RNN cells to behave as a single cell.

    Used to implement efficient stacked RNNs.

    # Arguments
        cells: List of RNN cell instances.

    # Examples

    ```python
        cells = [
            keras.layers.LSTMCell(output_dim),
            keras.layers.LSTMCell(output_dim),
            keras.layers.LSTMCell(output_dim),
        ]

        inputs = keras.Input((timesteps, input_dim))
        x = keras.layers.RNN(cells)(inputs)
    ```
    """

    def __init__(self, cells, **kwargs):
        for cell in cells:
            if not hasattr(cell, 'call'):
                raise ValueError('All cells must have a `call` method. '
                                 'received cells:', cells)
            if not hasattr(cell, 'state_size'):
                raise ValueError('All cells must have a '
                                 '`state_size` attribute. '
                                 'received cells:', cells)
        self.cells = cells
        super(StackedRNNCells, self).__init__(**kwargs)

    @property
    def state_size(self):
        # States are a flat list
        # in reverse order of the cell stack.
        # This allows to preserve the requirement
        # `stack.state_size[0] == output_dim`.
        # e.g. states of a 2-layer LSTM would be
        # `[h2, c2, h1, c1]`
        # (assuming one LSTM has states [h, c])
        state_size = []
        for cell in self.cells[::-1]:
            if hasattr(cell.state_size, '__len__'):
                state_size += list(cell.state_size)
            else:
                state_size.append(cell.state_size)
        return tuple(state_size)

    def call(self, inputs, states, constants=None, **kwargs):
        # Recover per-cell states.
        nested_states = []
        for cell in self.cells[::-1]:
            if hasattr(cell.state_size, '__len__'):
                nested_states.append(states[:len(cell.state_size)])
                states = states[len(cell.state_size):]
            else:
                nested_states.append([states[0]])
                states = states[1:]
        nested_states = nested_states[::-1]

        # Call the cells in order and store the returned states.
        new_nested_states = []
        for cell, states in zip(self.cells, nested_states):
            if has_arg(cell.call, 'constants'):
                inputs, states = cell.call(inputs, states,
                                           constants=constants,
                                           **kwargs)
            else:
                inputs, states = cell.call(inputs, states, **kwargs)
            new_nested_states.append(states)

        # Format the new states as a flat list
        # in reverse cell order.
        states = []
        for cell_states in new_nested_states[::-1]:
            states += cell_states
        return inputs, states

    def build(self, input_shape):
        if isinstance(input_shape, list):
            constants_shape = input_shape[1:]
            input_shape = input_shape[0]
        for cell in self.cells:
            if isinstance(cell, Layer):
                if has_arg(cell.call, 'constants'):
                    cell.build([input_shape] + constants_shape)
                else:
                    cell.build(input_shape)
            if hasattr(cell.state_size, '__len__'):
                output_dim = cell.state_size[0]
            else:
                output_dim = cell.state_size
            input_shape = (input_shape[0], output_dim)
        self.built = True

    def get_config(self):
        cells = []
        for cell in self.cells:
            cells.append({'class_name': cell.__class__.__name__,
                          'config': cell.get_config()})
        config = {'cells': cells}
        base_config = super(StackedRNNCells, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        cells = []
        for cell_config in config.pop('cells'):
            cells.append(deserialize_layer(cell_config,
                                           custom_objects=custom_objects))
        return cls(cells, **config)

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.non_trainable_weights
        if not self.trainable:
            trainable_weights = []
            for cell in self.cells:
                if isinstance(cell, Layer):
                    trainable_weights += cell.trainable_weights
            return trainable_weights + weights
        return weights

    def get_weights(self):
        """Retrieves the weights of the model.

        # Returns
            A flat list of Numpy arrays.
        """
        weights = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                weights += cell.weights
        return K.batch_get_value(weights)

    def set_weights(self, weights):
        """Sets the weights of the model.

        # Arguments
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.get_weights()`.
        """
        tuples = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                num_param = len(cell.weights)
                weights = weights[:num_param]
                for sw, w in zip(cell.weights, weights):
                    tuples.append((sw, w))
                weights = weights[num_param:]
        K.batch_set_value(tuples)

    @property
    def losses(self):
        losses = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                cell_losses = cell.losses
                losses += cell_losses
        return losses

    def get_losses_for(self, inputs=None):
        losses = []
        for cell in self.cells:
            if isinstance(cell, Layer):
                cell_losses = cell.get_losses_for(inputs)
                losses += cell_losses
        return losses


class RNN(Layer):
    """Base class for recurrent layers.

    # Arguments
        cell: A RNN cell instance. A RNN cell is a class that has:
            - a `call(input_at_t, states_at_t)` method, returning
                `(output_at_t, states_at_t_plus_1)`. The call method of the
                cell can also take the optional argument `constants`, see
                section "Note on passing external constants" below.
            - a `state_size` attribute. This can be a single integer
                (single state) in which case it is
                the size of the recurrent state
                (which should be the same as the size of the cell output).
                This can also be a list/tuple of integers
                (one size per state). In this case, the first entry
                (`state_size[0]`) should be the same as
                the size of the cell output.
            It is also possible for `cell` to be a list of RNN cell instances,
            in which cases the cells get stacked on after the other in the RNN,
            implementing an efficient stacked RNN.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively,
            the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)

    # Input shape
        3D tensor with shape `(batch_size, timesteps, input_dim)`.

    # Output shape
        - if `return_state`: a list of tensors. The first tensor is
            the output. The remaining tensors are the last states,
            each with shape `(batch_size, units)`.
        - if `return_sequences`: 3D tensor with shape
            `(batch_size, timesteps, units)`.
        - else, 2D tensor with shape `(batch_size, units)`.

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.

    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch. This assumes a one-to-one mapping
        between samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                if sequential model:
                  `batch_input_shape=(...)` to the first layer in your model.
                else for functional model with 1 or more Input layers:
                  `batch_shape=(...)` to all the first layers in your model.
                This is the expected shape of your inputs
                *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.
            - specify `shuffle=False` when calling fit().

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.

    # Note on specifying the initial state of RNNs
        You can specify the initial state of RNN layers symbolically by
        calling them with the keyword argument `initial_state`. The value of
        `initial_state` should be a tensor or list of tensors representing
        the initial state of the RNN layer.

        You can specify the initial state of RNN layers numerically by
        calling `reset_states` with the keyword argument `states`. The value of
        `states` should be a numpy array or list of numpy arrays representing
        the initial state of the RNN layer.

    # Note on passing external constants to RNNs
        You can pass "external" constants to the cell using the `constants`
        keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
        requires that the `cell.call` method accepts the same keyword argument
        `constants`. Such constants can be used to condition the cell
        transformation on additional static inputs (not changing over time),
        a.k.a. an attention mechanism.

    # Examples

    ```python
        # First, let's define a RNN Cell, as a layer subclass.

        class MinimalRNNCell(keras.layers.Layer):

            def __init__(self, units, **kwargs):
                self.units = units
                self.state_size = units
                super(MinimalRNNCell, self).__init__(**kwargs)

            def build(self, input_shape):
                self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                              initializer='uniform',
                                              name='kernel')
                self.recurrent_kernel = self.add_weight(
                    shape=(self.units, self.units),
                    initializer='uniform',
                    name='recurrent_kernel')
                self.built = True

            def call(self, inputs, states):
                prev_output = states[0]
                h = K.dot(inputs, self.kernel)
                output = h + K.dot(prev_output, self.recurrent_kernel)
                return output, [output]

        # Let's use this cell in a RNN layer:

        cell = MinimalRNNCell(32)
        x = keras.Input((None, 5))
        layer = RNN(cell)
        y = layer(x)

        # Here's how to use the cell to build a stacked RNN:

        cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
        x = keras.Input((None, 5))
        layer = RNN(cells)
        y = layer(x)
    ```
    """

    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if isinstance(cell, (list, tuple)):
            cell = StackedRNNCells(cell)
        if not hasattr(cell, 'call'):
            raise ValueError('`cell` should have a `call` method. '
                             'The RNN was passed:', cell)
        if not hasattr(cell, 'state_size'):
            raise ValueError('The RNN cell should have '
                             'an attribute `state_size` '
                             '(tuple of integers, '
                             'one integer per RNN state).')
        super(RNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = None

    @property
    def states(self):
        if self._states is None:
            if isinstance(self.cell.state_size, int):
                num_states = 1
            else:
                num_states = len(self.cell.state_size)
            return [None for _ in range(num_states)]
        return self._states

    @states.setter
    def states(self, states):
        self._states = states

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]
        output_dim = state_size[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], output_dim)
        else:
            output_shape = (input_shape[0], output_dim)

        if self.return_state:
            state_shape = [(input_shape[0], dim) for dim in state_size]
            return [output_shape] + state_shape
        else:
            return output_shape

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        input_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_dim))

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An `initial_state` was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'however `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(RNN, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(RNN, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(RNN, self).__call__(inputs, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps in [None, 1]:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined or equal to 1. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the batch size by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros((batch_size, dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros((batch_size, self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros((batch_size, dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros((batch_size, self.cell.state_size)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != (batch_size, dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'return_state': self.return_state,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll}
        if self._num_constants is not None:
            config['num_constants'] = self._num_constants

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'),
                                 custom_objects=custom_objects)
        num_constants = config.pop('num_constants', None)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.cell, Layer):
            return self.cell.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.cell, Layer):
            if not self.trainable:
                return self.cell.weights
            return self.cell.non_trainable_weights
        return []

    @property
    def losses(self):
        layer_losses = super(RNN, self).losses
        if isinstance(self.cell, Layer):
            return self.cell.losses + layer_losses
        return layer_losses

    def get_losses_for(self, inputs=None):
        if isinstance(self.cell, Layer):
            cell_losses = self.cell.get_losses_for(inputs)
            return cell_losses + super(RNN, self).get_losses_for(inputs)
        return super(RNN, self).get_losses_for(inputs)


class SimpleRNNCell(Layer):
    """Cell class for SimpleRNN.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(SimpleRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.state_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(prev_output),
                self.recurrent_dropout,
                training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask
        output = h + K.dot(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(RNN):
    """Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
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
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            warnings.warn('The `implementation` argument '
                          'in `SimpleRNN` has been deprecated. '
                          'Please remove it from your layer call.')
        dropout = 0. if dropout is None else dropout
        recurrent_dropout = 0. if recurrent_dropout is None else recurrent_dropout
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = SimpleRNNCell(units,
                             activation=activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout)
        super(SimpleRNN, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(SimpleRNN, self).call(inputs,
                                           mask=mask,
                                           training=training,
                                           initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


class GRUCell(Layer):
    """Cell class for the GRU layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (CuDNN compatible).
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 reset_after=False,
                 **kwargs):
        super(GRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 3),
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

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])
        else:
            self.bias = None

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            # bias for inputs
            self.input_bias_z = self.input_bias[:self.units]
            self.input_bias_r = self.input_bias[self.units: self.units * 2]
            self.input_bias_h = self.input_bias[self.units * 2:]
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.units]
                self.recurrent_bias_r = self.recurrent_bias[self.units: self.units * 2]
                self.recurrent_bias_h = self.recurrent_bias[self.units * 2:]
        else:
            self.input_bias_z = None
            self.input_bias_r = None
            self.input_bias_h = None
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = K.dot(inputs_z, self.kernel_z)
            x_r = K.dot(inputs_r, self.kernel_r)
            x_h = K.dot(inputs_h, self.kernel_h)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.input_bias_z)
                x_r = K.bias_add(x_r, self.input_bias_r)
                x_h = K.bias_add(x_h, self.input_bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
            recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, self.recurrent_bias_z)
                recurrent_r = K.bias_add(recurrent_r, self.recurrent_bias_r)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)

            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, self.input_bias)
            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            x_h = matrix_x[:, 2 * self.units:]

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1,
                                     self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:]
            else:
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:])

            hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(RNN):
    """Gated Recurrent Unit - Cho et al. 2014.

    There are two variants. The default one is based on 1406.1078v3 and
    has reset gate applied to hidden state before matrix multiplication. The
    other one is based on original 1406.1078v1 and has the order reversed.

    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. Use `'reset_after'=True` and
    `recurrent_activation='sigmoid'`.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
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
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (CuDNN compatible).

    # References
        - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=2`.'
                          'Please update your layer call.')
        dropout = 0. if dropout is None else dropout
        recurrent_dropout = 0. if recurrent_dropout is None else recurrent_dropout
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = GRUCell(units,
                       activation=activation,
                       recurrent_activation=recurrent_activation,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       recurrent_initializer=recurrent_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       recurrent_regularizer=recurrent_regularizer,
                       bias_regularizer=bias_regularizer,
                       kernel_constraint=kernel_constraint,
                       recurrent_constraint=recurrent_constraint,
                       bias_constraint=bias_constraint,
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       implementation=implementation,
                       reset_after=reset_after)
        super(GRU, self).__init__(cell,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(GRU, self).call(inputs,
                                     mask=mask,
                                     training=training,
                                     initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    @property
    def reset_after(self):
        return self.cell.reset_after

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(GRU, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


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
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
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
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
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
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
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
        dropout_w_a: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation in the attended context.
        dropout_W_a: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state in the attention mechanism.
        dropout_U_a: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input in the attention mechanism.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
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

        assert len(input_shape) >= 2, 'You should pass two inputs to AttLSTMCond ' \
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

        self.attention_context_wa = self.add_weight(
            shape=(self.att_units,),
            name='attention_context_wa',
            initializer=self.attention_context_wa_initializer,
            regularizer=self.attention_context_wa_regularizer,
            constraint=self.attention_context_wa_constraint)

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
        bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
        self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                       name='bias_ca',
                                       initializer=self.bias_ca_initializer,
                                       regularizer=self.bias_ca_regularizer,
                                       constraint=self.bias_ca_constraint)

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

        self.attention_context_wa = self.add_weight(
            shape=(self.att_units,),
            name='attention_context_wa',
            initializer=self.attention_context_wa_initializer,
            regularizer=self.attention_context_wa_regularizer,
            constraint=self.attention_context_wa_constraint)

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
        bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
        self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                       name='bias_ca',
                                       initializer=self.bias_ca_initializer,
                                       regularizer=self.bias_ca_regularizer,
                                       constraint=self.bias_ca_constraint)
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

        self.attention_context_wa = self.add_weight(
            shape=(self.att_units,),
            name='attention_context_wa',
            initializer=self.attention_context_wa_initializer,
            regularizer=self.attention_context_wa_regularizer,
            constraint=self.attention_context_wa_constraint)

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
        bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
        self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                       name='bias_ca',
                                       initializer=self.bias_ca_initializer,
                                       regularizer=self.bias_ca_regularizer,
                                       constraint=self.bias_ca_constraint)
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


class LSTMCell(Layer):
    """Cell class for the LSTM layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 **kwargs):
        super(LSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.implementation = implementation
        self.state_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
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

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
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

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
                                                      self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
                                                      self.recurrent_kernel_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                            self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                                                      self.recurrent_kernel_o))
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(LSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTM(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
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
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    # References
        - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=2`.'
                          'Please update your layer call.')
        dropout = 0. if dropout is None else dropout
        recurrent_dropout = 0. if recurrent_dropout is None else recurrent_dropout
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = LSTMCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        super(LSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(LSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(LSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)


def _standardize_args(inputs, initial_state, constants, num_constants):
    """Standardize `__call__` to a single list of tensor inputs.

    When running a model loaded from file, the input tensors
    `initial_state` and `constants` can be passed to `RNN.__call__` as part
    of `inputs` instead of by the dedicated keyword arguments. This method
    makes sure the arguments are separated and that `initial_state` and
    `constants` are lists of tensors (or None).

    # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None

    # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
    """
    if isinstance(inputs, list):
        assert initial_state is None and constants is None
        if num_constants is not None:
            constants = inputs[-num_constants:]
            inputs = inputs[:-num_constants]
        if len(inputs) > 1:
            initial_state = inputs[1:]
        inputs = inputs[0]

    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    initial_state = to_list_or_none(initial_state)
    constants = to_list_or_none(constants)

    return inputs, initial_state, constants


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
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
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
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
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
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 forget_bias_init='one',
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
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
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.forget_bias_initializer = initializers.get(forget_bias_init)

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

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
                  'return_states': self.return_states,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'conditional_initializer': initializers.serialize(self.conditional_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'bias_ba_initializer': initializers.serialize(self.bias_ba_initializer),
                  'bias_ca_initializer': initializers.serialize(self.bias_ca_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'conditional_regularizer': regularizers.serialize(self.conditional_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'bias_ba_regularizer': regularizers.serialize(self.bias_ba_regularizer),
                  'bias_ca_regularizer': regularizers.serialize(self.bias_ca_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'conditional_constraint': constraints.serialize(self.conditional_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'bias_ba_constraint': constraints.serialize(self.bias_ba_constraint),
                  'bias_ca_constraint': constraints.serialize(self.bias_ca_constraint),
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
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
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
        dropout_w_a: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation in the attended context.
        dropout_W_a: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state in the attention mechanism.
        dropout_U_a: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input in the attention mechanism.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
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
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
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

        self.attention_context_wa = self.add_weight(
            shape=(self.att_units,),
            name='attention_context_wa',
            initializer=self.attention_context_wa_initializer,
            regularizer=self.attention_context_wa_regularizer,
            constraint=self.attention_context_wa_constraint)

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
        bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
        self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                       name='bias_ca',
                                       initializer=self.bias_ca_initializer,
                                       regularizer=self.bias_ca_regularizer,
                                       constraint=self.bias_ca_constraint)

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
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
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

        if self.attention_mode != 'dot':
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
        if self.attention_mode != 'dot':
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

        self.attention_context_wa = self.add_weight(
            shape=(self.att_units,),
            name='attention_context_wa',
            initializer=self.attention_context_wa_initializer,
            regularizer=self.attention_context_wa_regularizer,
            constraint=self.attention_context_wa_constraint)

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
            bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           name='bias_ca',
                                           initializer=self.bias_ca_initializer,
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)

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
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
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
        - [Egocentric Video Description based on Temporally-Linked Sequences](https://arxiv.org/abs/1704.02163)
    """

    def __init__(self, units,
                 att_units1=0,
                 att_units2=0,
                 return_states=False,
                 activation='tanh',
                 attention_mode='add',
                 recurrent_activation='sigmoid',
                 return_extra_variables=False,
                 attend_on_both=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_initializer2='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_recurrent_initializer2='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_initializer2='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 attention_context_wa_initializer2='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 bias_initializer2='zeros',
                 bias_ba_initializer='zeros',
                 bias_ba_initializer2='zeros',
                 bias_ca_initializer='zero',
                 bias_ca_initializer2='zero',
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_regularizer=None,
                 kernel_regularizer2=None,
                 conditional_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 bias_regularizer2=None,
                 attention_context_regularizer=None,
                 attention_context_regularizer2=None,
                 attention_context_wa_regularizer=None,
                 attention_context_wa_regularizer2=None,
                 attention_recurrent_regularizer=None,
                 attention_recurrent_regularizer2=None,
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

        self.attention_context_wa = self.add_weight(shape=(self.att_units1,),
                                                    initializer=self.attention_context_wa_initializer,
                                                    name='attention_context_wa',
                                                    regularizer=self.attention_context_wa_regularizer,
                                                    constraint=self.attention_context_wa_constraint)

        self.bias_ba = self.add_weight(shape=(self.att_units1,),
                                       initializer=self.bias_ba_initializer,
                                       name='bias_ba',
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        bias_ca_shape = self.context1_steps if self.context1_steps is None else (self.context1_steps,)
        self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                       initializer=self.bias_ca_initializer,
                                       name='bias_ca',
                                       regularizer=self.bias_ca_regularizer,
                                       constraint=self.bias_ca_constraint)

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

            self.attention_context_wa2 = self.add_weight(shape=(self.att_units2,),
                                                         initializer=self.attention_context_wa_initializer2,
                                                         name='attention_context_wa2',
                                                         regularizer=self.attention_context_wa_regularizer2,
                                                         constraint=self.attention_context_wa_constraint2)

            self.bias_ba2 = self.add_weight(shape=(self.att_units2,),
                                            initializer=self.bias_ba_initializer2,
                                            name='bias_ba2',
                                            regularizer=self.bias_ba_regularizer2,
                                            constraint=self.bias_ba_constraint2)
            bias_ca_shape = self.context2_steps if self.context2_steps is None else (self.context2_steps,)
            self.bias_ca2 = self.add_weight(shape=bias_ca_shape,
                                            initializer=self.bias_ca_initializer2,
                                            name='bias_ca2',
                                            regularizer=self.bias_ca_regularizer2,
                                            constraint=self.bias_ca_constraint2)

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

        ctx_1, alphas1 = compute_attention(h_tm1, pctx_1, context, att_dp_mask, self.attention_recurrent_kernel,
                                           self.attention_context_wa, self.bias_ca, mask_context1,
                                           attention_mode=self.attention_mode)

        if self.attend_on_both:
            if K.ndim(mask_context2) > 1:  # Mask the context2 (only if necessary)
                pctx_2 = mask_context2[:, :, None] * pctx_2
                context2 = mask_context2[:, :, None] * context2

            # Attention model 2 (see Formulation in class header)
            ctx_2, alphas2 = compute_attention(h_tm1, pctx_1, context, att_dp_mask2, self.attention_recurrent_kernel2,
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
                z = K.bias_add(z_, self.bias2)
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
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
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
        - [Egocentric Video Description based on Temporally-Linked Sequences](https://arxiv.org/abs/1704.02163)
    """

    def __init__(self, units,
                 att_units1=0,
                 att_units2=0,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 return_extra_variables=False,
                 attend_on_both=False,
                 use_bias=True,
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_initializer='glorot_uniform',
                 kernel_initializer2='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_recurrent_initializer2='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_initializer2='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 attention_context_wa_initializer2='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 bias_initializer2='zeros',
                 bias_ba_initializer='zeros',
                 bias_ba_initializer2='zeros',
                 bias_ca_initializer='zero',
                 bias_ca_initializer2='zero',
                 kernel_regularizer=None,
                 kernel_regularizer2=None,
                 conditional_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 bias_regularizer2=None,
                 attention_context_regularizer=None,
                 attention_context_regularizer2=None,
                 attention_context_wa_regularizer=None,
                 attention_context_wa_regularizer2=None,
                 attention_recurrent_regularizer=None,
                 attention_recurrent_regularizer2=None,
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

        self.attention_context_wa = self.add_weight(shape=(self.att_units1,),
                                                    initializer=self.attention_context_wa_initializer,
                                                    name='attention_context_wa',
                                                    regularizer=self.attention_context_wa_regularizer,
                                                    constraint=self.attention_context_wa_constraint)

        self.bias_ba = self.add_weight(shape=(self.att_units1,),
                                       initializer=self.bias_ba_initializer,
                                       name='bias_ba',
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        bias_ca_shape = self.context1_steps if self.context1_steps is None else (self.context1_steps,)
        self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                       initializer=self.bias_ca_initializer,
                                       name='bias_ca',
                                       regularizer=self.bias_ca_regularizer,
                                       constraint=self.bias_ca_constraint)

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

            self.attention_context_wa2 = self.add_weight(shape=(self.att_units2,),
                                                         initializer=self.attention_context_wa_initializer2,
                                                         name='attention_context_wa2',
                                                         regularizer=self.attention_context_wa_regularizer2,
                                                         constraint=self.attention_context_wa_constraint2)

            self.bias_ba2 = self.add_weight(shape=(self.att_units2,),
                                            initializer=self.bias_ba_initializer2,
                                            name='bias_ba2',
                                            regularizer=self.bias_ba_regularizer2,
                                            constraint=self.bias_ba_constraint2)
            bias_ca_shape = self.context2_steps if self.context2_steps is None else (self.context2_steps,)
            self.bias_ca2 = self.add_weight(shape=bias_ca_shape,
                                            initializer=self.bias_ca_initializer2,
                                            name='bias_ca2',
                                            regularizer=self.bias_ca_regularizer2,
                                            constraint=self.bias_ca_constraint2)

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
        ctx_, alphas = compute_attention(h_, pctx_1, context, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, mask_context1,
                                         attention_mode=self.attention_mode)

        if self.attend_on_both:
            if K.ndim(mask_context2) > 1:  # Mask the context2 (only if necessary)
                pctx_2 = mask_context2[:, :, None] * pctx_2
                context2 = mask_context2[:, :, None] * context2
            # Attention model 2 (see Formulation in class header)
            ctx_2, alphas2 = compute_attention(h_, pctx_1, context, att_dp_mask2, self.attention_recurrent_kernel2,
                                               self.attention_context_wa2, self.bias_ca2, mask_context2,
                                               attention_mode=self.attention_mode)
        else:
            ctx_2 = context2
            alphas2 = mask_context2

        # LSTM_2
        z = x + \
            K.dot(h_ * rec_dp_mask[0], self.recurrent_kernel) + \
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
        c = f * c_ + i * self.activation(z2)
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
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
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
        - [Egocentric Video Description based on Temporally-Linked Sequences](https://arxiv.org/abs/1704.02163)
    """

    def __init__(self,
                 units,
                 att_units1=0,
                 att_units2=0,
                 att_units3=0,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 return_states=False,
                 return_extra_variables=False,
                 attend_on_both=False,
                 use_bias=True,
                 unit_forget_bias=True,
                 mask_value=0.,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 init_att='glorot_uniform',
                 kernel_initializer='glorot_uniform',
                 kernel_initializer2='glorot_uniform',
                 kernel_initializer3='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_recurrent_initializer2='glorot_uniform',
                 attention_recurrent_initializer3='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_initializer2='glorot_uniform',
                 attention_context_initializer3='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 attention_context_wa_initializer2='glorot_uniform',
                 attention_context_wa_initializer3='glorot_uniform',
                 recurrent_initializer='orthogonal',
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
                 conditional_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 bias_regularizer2=None,
                 bias_regularizer3=None,
                 attention_context_regularizer=None,
                 attention_context_regularizer2=None,
                 attention_context_regularizer3=None,
                 attention_context_wa_regularizer=None,
                 attention_context_wa_regularizer2=None,
                 attention_context_wa_regularizer3=None,
                 attention_recurrent_regularizer=None,
                 attention_recurrent_regularizer2=None,
                 attention_recurrent_regularizer3=None,
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
        assert len(input_shape) >= 4 or 'You should pass four inputs to AttLSTMCond2Inputs ' \
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
                                                                                                                  'to AttLSTMCond2Inputs'
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

        self.attention_context_wa = self.add_weight(shape=(self.att_units1,),
                                                    initializer=self.attention_context_wa_initializer,
                                                    name='attention_context_wa',
                                                    regularizer=self.attention_context_wa_regularizer,
                                                    constraint=self.attention_context_wa_constraint)

        self.bias_ba = self.add_weight(shape=(self.att_units1,),
                                       initializer=self.bias_ba_initializer,
                                       name='bias_ba',
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        bias_ca_shape = self.context1_steps if self.context1_steps is None else (self.context1_steps,)
        self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                       initializer=self.bias_ca_initializer,
                                       name='bias_ca',
                                       regularizer=self.bias_ca_regularizer,
                                       constraint=self.bias_ca_constraint)

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

        if self.attend_on_both:
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

        if self.consume_less == 'gpu':

            self.T = self.add_weight(shape=(self.context1_dim, 4 * self.units),
                                     initializer=self.init,
                                     name='{}_T'.format(self.name),
                                     regularizer=self.W_regularizer)
            self.W = self.add_weight(shape=(self.context2_dim, 4 * self.units),
                                     initializer=self.init,
                                     name='{}_W'.format(self.name),
                                     regularizer=self.W_regularizer)
            self.S = self.add_weight(shape=(self.context3_dim, 4 * self.units),
                                     initializer=self.init,
                                     name='{}_S'.format(self.name),
                                     regularizer=self.S_regularizer)

            self.U = self.add_weight(shape=(self.units, 4 * self.units),
                                     initializer=self.inner_init,
                                     name='{}_U'.format(self.name),
                                     regularizer=self.U_regularizer)
            self.V = self.add_weight(shape=(self.input_dim, 4 * self.units),
                                     initializer=self.init,
                                     name='{}_V'.format(self.name),
                                     regularizer=self.V_regularizer)

            '''
            def b_reg(shape, name=None):
                return K.variable(np.hstack((np.zeros(self.units),
                                             K.get_value(self.forget_bias_init((self.units,))),
                                             np.zeros(self.units),
                                             np.zeros(self.units))),
                                  name='{}_b'.format(self.name))
            '''

            self.b = self.add_weight(shape=(self.units * 4,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)

            """
            self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, self.ca,  # AttModel parameters
                                      self.S,
                                      self.T,
                                      self.W,
                                      self.U,
                                      self.V,
                                      self.b]
            if self.attend_on_both:
                self.trainable_weights += [self.wa2, self.Wa2, self.Ua2, self.ba2, self.ca2,  # AttModel2 parameters)
                                           self.wa3, self.Wa3, self.Ua3, self.ba3, self.ca3  # AttModel3 parameters)
                                           ]
            """

        else:
            raise NotImplementedError

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
            ctx_2, alphas2 = compute_attention(h_tm1, pctx_, context, B_Wa2, self.Wa2,
                                               self.wa2, self.ca2, mask_context2,
                                               attention_mode=self.attention_mode)
            # Attention model 3 (see Formulation in class header)
            ctx_3, alphas3 = compute_attention(h_tm1, pctx_, context, B_Wa3, self.Wa3,
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
