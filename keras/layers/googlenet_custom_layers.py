from keras.layers.core import Layer


class LRN(Layer):
    """Local Response Normalization (LRN).
    The local response normalization layer performs a kind of "lateral inhibition"
    by normalizing over local input regions.
    In ACROSS_CHANNELS mode, the local regions extend across nearby channels,
    but have no spatial extent (i.e., they have shape local_size x 1 x 1).
    In WITHIN_CHANNEL mode, the local regions extend spatially, but are in separate channels
    (i.e., they have shape 1 x local_size x local_size). Each input value is divided by (1+(\alpha /n)\sum_i{x_i^2)}^\beta,
    where n is the size of each local region, and the sum is taken over the region centered at
    that value (zero padding is added where necessary).

    # Arguments
        alpha: scaling parameter.
        k: offset for the scale
        beta: the exponent.
        n: local_size
    """

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        import theano.tensor as T
        b, ch, r, c = x.shape
        half_n = self.n // 2  # half the local region
        input_sqr = T.sqr(x)  # square the input
        extra_channels = T.alloc(0., b, ch + 2 * half_n, r,
                                 c)  # make an empty tensor with zero pads along channel dimension
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n + ch, :, :],
                                    input_sqr)  # set the center to be the squared input
        scale = self.k  # offset for the scale
        norm_alpha = self.alpha / self.n  # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):
    """PoolHelper.
    Pooling utility.
    """
    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, 1:, 1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
