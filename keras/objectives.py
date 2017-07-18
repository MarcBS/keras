"""Legacy objectives module.

Only kept for backwards API compatibility.
"""
from __future__ import absolute_import
from .losses import *


def log_diff(args):
    ''' log prob difference between a GT and a hypothesis
    '''
    y_pred, y_true, h_pred, h_true = args
    p_y_x = K.mean(K.categorical_crossentropy(y_pred, y_true))
    p_h_x = K.mean(K.categorical_crossentropy(h_pred, h_true))
    l = p_y_x - p_h_x
    return l


def y_true(y_true, y_pred):
    '''Returns the label (y_true)
    '''
    return y_true

