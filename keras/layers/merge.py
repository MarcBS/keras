"""Layers that can merge several inputs into one.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..engine.base_layer import Layer
from .. import backend as K


class _Merge(Layer):
    """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.

        # Arguments
            shape1: tuple or None. Shape of the first tensor
            shape2: tuple or None. Shape of the second tensor

        # Returns
            expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.

        # Raises
            ValueError: if shape1 and shape2 are not compatible for
                element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError('Operands could not be broadcast '
                                     'together with shapes ' +
                                     str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) > 1:
            raise ValueError('Can not merge tensors with different '
                             'batch sizes. Got tensors with shapes : ' +
                             str(input_shape))
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape,
                                                                  shape)
        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(K.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = K.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = K.expand_dims(x, 1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = K.ndim(x)
                    if x_ndim is None:
                        x_shape = K.shape(x)
                        batch_size = x_shape[0]
                        new_shape = K.concatenate([x_shape[1:],
                                                   K.expand_dims(batch_size)])
                        x_transposed = K.reshape(x, K.stack([batch_size,
                                                             K.prod(x_shape[1:])]))
                        x_transposed = K.permute_dimensions(x_transposed, (1, 0))
                        x_transposed = K.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(K.permute_dimensions(x, dims))
                        transposed = True
                    else:
                        # We don't transpose inputs if they are
                        # 1D vectors or scalars.
                        reshaped_inputs.append(x)
                y = self._merge_function(reshaped_inputs)
                y_ndim = K.ndim(y)
                if transposed:
                    # If inputs have been transposed,
                    # we have to transpose the output too.
                    if y_ndim is None:
                        y_shape = K.shape(y)
                        y_ndim = K.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = K.concatenate([K.expand_dims(batch_size),
                                                   y_shape[:y_ndim - 1]])
                        y = K.reshape(y, (-1, batch_size))
                        y = K.permute_dimensions(y, (1, 0))
                        y = K.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = K.permute_dimensions(y, dims)
                return y
        else:
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape,
                                                                  shape)
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        masks = [K.expand_dims(K.cast(m, K.floatx()), 0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


class Add(_Merge):
    """Layer that adds a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # equivalent to added = keras.layers.add([x1, x2])
        added = keras.layers.Add()([x1, x2])

        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output


class Subtract(_Merge):
    """Layer that subtracts two inputs.

    It takes as input a list of tensors of size 2,
    both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
    also of the same shape.

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # Equivalent to subtracted = keras.layers.subtract([x1, x2])
        subtracted = keras.layers.Subtract()([x1, x2])

        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """

    def build(self, input_shape):
        super(Subtract, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Subtract` layer should be called '
                             'on exactly 2 inputs')
        return inputs[0] - inputs[1]


class Multiply(_Merge):
    """Layer that multiplies (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output *= inputs[i]
        return output


class Average(_Merge):
    """Layer that averages a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output / len(inputs)


class Maximum(_Merge):
    """Layer that computes the maximum (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = K.maximum(output, inputs[i])
        return output


class Minimum(_Merge):
    """Layer that computes the minimum (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = K.minimum(output, inputs[i])
        return output


class Concatenate(_Merge):
    """Layer that concatenates a list of inputs.

    It takes as input a list of tensors,
    all of the same shape except for the concatenation axis,
    and returns a single tensor, the concatenation of all inputs.

    # Arguments
        axis: Axis along which to concatenate.
        cropping: list of axies to crop on.
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, axis=-1, cropping=None, **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False

        if cropping is not None:
            # If cropping is enabled, don't crop on the selected axis
            cropping = list(cropping)
            cropping[axis] = None
        self.cropping = cropping

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of at least 2 inputs')
        if all([shape is None for shape in input_shape]):
            return

        input_shapes = autocrop_array_shapes(input_shape, self.cropping)
        reduced_inputs_shapes = [list(shape) for shape in input_shapes]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('A `Concatenate` layer requires '
                             'inputs with matching shapes '
                             'except for the concat axis. '
                             'Got inputs shapes: %s' % (input_shapes))

    def _merge_function(self, inputs):
        inputs = autocrop(inputs, self.cropping)
        return K.concatenate(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of inputs.')
        input_shapes = autocrop_array_shapes(input_shape, self.cropping)
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        # Make a list of masks while making sure
        # the dimensionality of each mask
        # is the same as the corresponding input.
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                # Input is unmasked. Append all 1s to masks,
                masks.append(K.ones_like(input_i, dtype='bool'))
            elif K.ndim(mask_i) < K.ndim(input_i):
                # Mask is smaller than the input, expand it
                masks.append(K.expand_dims(mask_i))
            else:
                masks.append(mask_i)
        concatenated = K.concatenate(masks, axis=self.axis)
        return K.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {
            'axis': self.axis, 'cropping': self.cropping
        }
        base_config = super(Concatenate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dot(_Merge):
    """Layer that computes a dot product between samples in two tensors.

    E.g. if applied to a list of two tensors `a` and `b` of shape `(batch_size, n)`,
    the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.

    # Arguments
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    """

    def __init__(self, axes, normalize=False, **kwargs):
        super(Dot, self).__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError('Invalid type for `axes` - '
                                'should be a list or an int.')
            if len(axes) != 2:
                raise ValueError('Invalid format for `axes` - '
                                 'should contain two elements.')
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError('Invalid format for `axes` - '
                                 'list elements should be "int".')
        self.axes = axes
        self.normalize = normalize
        self.supports_masking = True
        self._reshape_required = False

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                'Layer shapes: %s, %s' % (shape1, shape2))

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on exactly 2 inputs')
        x1 = inputs[0]
        x2 = inputs[1]
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % K.ndim(inputs[i]))
                else:
                    axes.append(self.axes[i])
        if self.normalize:
            x1 = K.l2_normalize(x1, axis=axes[0])
            x2 = K.l2_normalize(x2, axis=axes[1])
        output = K.batch_dot(x1, x2, axes)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'axes': self.axes,
            'normalize': self.normalize,
        }
        base_config = super(Dot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Alias
Sum = Add


def add(inputs, **kwargs):
    """Functional interface to the `Add` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the sum of the inputs.

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        added = keras.layers.add([x1, x2])

        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """
    return Add(**kwargs)(inputs)


def subtract(inputs, **kwargs):
    """Functional interface to the `Subtract` layer.

    # Arguments
        inputs: A list of input tensors (exactly 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the difference of the inputs.

    # Examples

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        subtracted = keras.layers.subtract([x1, x2])

        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """
    return Subtract(**kwargs)(inputs)


def multiply(inputs, **kwargs):
    """Functional interface to the `Multiply` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise product of the inputs.
    """
    return Multiply(**kwargs)(inputs)


def average(inputs, **kwargs):
    """Functional interface to the `Average` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the average of the inputs.
    """
    return Average(**kwargs)(inputs)


def maximum(inputs, **kwargs):
    """Functional interface to the `Maximum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise maximum of the inputs.
    """
    return Maximum(**kwargs)(inputs)


def minimum(inputs, **kwargs):
    """Functional interface to the `Minimum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise minimum of the inputs.
    """
    return Minimum(**kwargs)(inputs)


def concatenate(inputs, axis=-1, **kwargs):
    """Functional interface to the `Concatenate` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the concatenation of the inputs alongside axis `axis`.
    """
    return Concatenate(axis=axis, **kwargs)(inputs)


def dot(inputs, axes, normalize=False, **kwargs):
    """Functional interface to the `Dot` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the dot product of the samples from the inputs.
    """
    return Dot(axes=axes, normalize=normalize, **kwargs)(inputs)


def autocrop(inputs, cropping):
    """
    Crops the given input arrays.

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

    Parameters
    ----------
    inputs : list of Theano expressions
        The input arrays in the form of a list of Theano expressions

    cropping : list of cropping modes
        Cropping modes, one for each axis. If length of `cropping` is less
        than the number of axes in the inputs, it is padded with `None`.
        If `cropping` is None, `input` is returned as is.

    Returns
    -------
    list of Theano expressions

        each expression is the cropped version of the corresponding input
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
                             "dimensionalities {1}.".format(len(inputs), [K.ndim(input) for input in inputs]))

        # Get the shape of each input
        shapes = [K.shape(input) for input in inputs]
        # Convert the shapes to a matrix expression
        shapes_tensor = K.as_tensor_variable(shapes)
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
                    raise ValueError('Unknown crop mode \'{0}\''.format(cr))

        return [input[slices] for input, slices in
                zip(inputs, slices_by_input)]


def autocrop_array_shapes(input_shapes, cropping):
    """
    Computes the shapes of the given arrays after auto-cropping is applied.

    For more information on cropping, see the :func:`autocrop` function
    documentation.

    Parameters
    ----------
    input_shapes : the shapes of input arrays prior to cropping in
        the form of a list of tuples

    cropping : a list of cropping modes, one for each axis. If length of
        `cropping` is less than the number of axes in the inputs, it is
        padded with `None`. If `cropping` is None, `input_shapes` is returned
        as is. For more information on their values and operation, see the
        :func:`autocrop` documentation.
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
                min_sh = None if any(x is None for x in sh) else min(sh)
                result.append([min_sh] * len(sh))
            else:
                raise ValueError('Unknown crop mode \'{0}\''.format(cr))
        return [tuple(sh) for sh in zip(*result)]
