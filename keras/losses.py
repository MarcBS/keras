from __future__ import absolute_import
import six
from . import backend as K
from .utils.generic_utils import deserialize_keras_object
from .utils.generic_utils import serialize_keras_object


# noinspection SpellCheckingInspection
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)


def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.

    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction. However, it may return NaNs if the
    intermediate value `cosh(y_pred - y_true)` is too large to be represented
    in the chosen precision.
    """
    def cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    return K.mean(K.log(cosh(y_pred - y_true)), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)

def log_diff(args):
    """
     Cross-entropy difference between a GT and a hypothesis
    :param args: y_pred, y_true, h_pred, h_true
    :return:
    """
    y_true, y_pred, h_true, h_pred = args
    p_y_x = K.mean(K.categorical_crossentropy(y_true, y_pred))
    p_h_x = K.mean(K.categorical_crossentropy(h_true, h_pred))
    cost_difference = p_y_x - p_h_x
    cost_difference = K.switch(cost_difference > -1., cost_difference, 0.)
    return cost_difference


def hybrid_log_diff(args):
    """
     Weighted Cross-entropy difference between a GT and a hypothesis
    :param args: y_pred, y_true, h_pred, h_true
    :return:
    """
    y_true, y_pred, h_true, h_pred, weight1, weight2, constant = args
    p_y_x =  K.mean(K.categorical_crossentropy(y_true, y_pred))
    p_h_x = K.mean(K.categorical_crossentropy(h_true, h_pred))
    cost_difference1 = p_y_x - p_h_x
    cost_difference1 = K.switch(cost_difference1 > -1., cost_difference1, 0.)

    cost_difference2 = p_y_x - constant
    cost_difference2 = K.switch(cost_difference2 > -1., cost_difference2, 0.)

    return weight1 * cost_difference1 + weight2 * cost_difference2


def weighted_log_diff(args):
    """
     Weighted Cross-entropy difference between a GT and a hypothesis
    :param args: y_pred, y_true, h_pred, h_true
    :return:
    """
    y_true, y_pred, h_true, h_pred, weight = args
    p_y_x =  K.mean(K.categorical_crossentropy(y_true, y_pred))
    p_h_x = K.mean(K.categorical_crossentropy(h_true, h_pred))
    return p_y_x - weight * p_h_x

def log_diff_plus_categorical_crossentropy(args):
    """
     Weighted cross-entropy difference between a GT and a hypothesis and weigthed log-diff
    :param args: y_pred, y_true, h_pred, h_true
    :return:
    """
    y_true, y_pred, h1_true, h1_pred, h2_true, h2_pred, weight = args
    p_y_x =  K.mean(K.categorical_crossentropy(y_true, y_pred))
    p_h1_x = K.mean(K.categorical_crossentropy(h1_true, h1_pred))
    p_h2_x = K.mean(K.categorical_crossentropy(h2_true, h2_pred))
    cost_difference = p_h1_x - p_h2_x
    #cost_difference = K.switch(cost_difference > -1., cost_difference, 0.)
    return weight * p_y_x + (1 - weight) * cost_difference


def linear_interpolation_categorical_crossentropy(args):
    y_true, y_pred, additional_metric, weight = args
    return K.mean(K.categorical_crossentropy(y_true, y_pred)) + weight * additional_metric


def y_true(y_true, y_pred):
    """
    Returns the label (y_true)
    :param y_true:
    :param y_pred:
    :return:
    """
    return y_true


def nce_correct_prob(y_pred, y_noise):
    """
    p(correct| x, y) used in NCE.
    :param y_pred: Model distribution (p_m(y|x; \theta))
    :param y_noise: Noisy distribution (p_n(y|x))
    :return: Probability that a given example is predicted to be a correct training (p(correct|x, y))
    """
    return y_pred / (y_pred + y_noise)


def noise_contrastive_loss(args):
    """
    :param y_distribution:
    :param y_noise:
    :return:
    """
    #TODO: We are assuming that |U_t| == |U_n|
    pred_true_model, y_true_model, pred_noise_model, y_true_noise_model = args
    y_distribution_pred = K.mean(K.categorical_crossentropy(y_true_model, pred_true_model))
    y_distribution_noise = K.mean(K.categorical_crossentropy(y_true_model, pred_noise_model))

    y_noise_pred = K.mean(K.categorical_crossentropy(y_true_noise_model, pred_true_model))
    y_noise_noise = K.mean(K.categorical_crossentropy(y_true_noise_model, pred_noise_model))

    return K.mean(K.log(nce_correct_prob(y_distribution_pred, y_distribution_noise))) +\
           K.mean(K.log(1 - nce_correct_prob(y_noise_pred, y_noise_noise)))



# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


def serialize(loss):
    return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
