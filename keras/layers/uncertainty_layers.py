# -*- coding: utf-8 -*-
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from .. import backend as K
from ..engine import Layer


class ConcatenateOutputWithSigma(Layer):
    """
    TODO: Write documentation.

    Arguments:
        output_dim: ...
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ConcatenateOutputWithSigma, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(1,),
                                      name='sigma',
                                      initializer='zeros',
                                      trainable=True)
        # Be sure to call this somewhere!
        super(ConcatenateOutputWithSigma, self).build(input_shape)

    def call(self, x):
        input_ones_matrix = ((K.abs(x) + 1) / (K.abs(x) + 1))
        batch_kernel = K.expand_dims(input_ones_matrix[:, 0], -1) * self.kernel
        return K.concatenate((x, batch_kernel), -1)

    def compute_output_shape(self, input_shape):
        return self.output_dim

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(ConcatenateOutputWithSigma, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
