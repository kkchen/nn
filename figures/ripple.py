import numpy as np

class Ripple:

    """Class to produce radial ripple function."""

    def __init__(self, extent, batchsize):
        self._extent = extent
        self._batchsize = batchsize

    def random_batch(self):
        """Get a random minibatch."""
        x = self._random_x()
        y = self._random_x()

        return x, y, self.f(x, y)

    @staticmethod
    def f(x, y):
        """The function to model, as x and y."""
        return Ripple.g(np.sqrt(x ** 2 + y ** 2))

    @staticmethod
    def g(r):
        """The function to model, as radius."""
        return np.cos(r) / (np.exp(0.4 * (r - 8)) + 1)

    def _random_x(self):
        """Get a random minibatch draw on x."""
        return np.random.uniform(-self._extent, self._extent, self._batchsize)
