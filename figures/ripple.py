#!/usr/bin/env python3

"""Train dense networks on a ripple function."""

import argparse
import warnings

# from matplotlib import cm
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from tensorflow import keras

class Ripple:

    """Class to produce radial ripple function."""

    def __init__(self, extent, batchsize):
        self._extent = extent
        self._batchsize = batchsize

    def random_batch():
        """Get a random minibatch."""
        x = self._random_x()
        y = self._random_x()

        return self.f(x, y)

    @staticmethod
    def f(x, y):
        """The function to model, as x and y."""
        return g(np.sqrt(x ** 2 + y ** 2))

    @staticmethod
    def g(r):
        """The function to model, as radius."""
        return np.cos(r) / (np.exp(0.4 * (r - 8)) + 1)

    def _random_x():
        """Get a random minibatch draw on x."""
        return np.random.uniform(-self._extent, self._extent, self._batchsize)


class KerasDense:

    """Class for training Keras dense model."""

    def __init__(self, args):
        self._layers = args.layers
        self._iterations = round(args.n_data / args.minibatch_size)
        self._minibatch_size = args.minibatch_size
        self._activation = args.activation
        self._loss = args.loss
        self._optimizer = args.optimizer

        self._model = keras.models.Sequential()
        self._ripple = Ripple(args.extent, self._minibatch_size)

    def build(self):
        """Build the model."""
        n_neurons = (
            (250,),
            (42, 20),
            (38, 18, 9),
            (36, 18, 9, 5)
        )

        self._model.add(
            keras.layers.Dense(
                n_neurons[self._layers-1][0],
                self._activation,
                input_shape=(2,)
            )
        )

        for layer in range(1, self._layers):
            self._model.add(
                keras.layers.Dense(n_neurons[self._layers-1][layer], self._activation)
            )

        self._model.add(keras.layers.Dense(1))
        self._model.summary()

    def train(self):
        """Train the model."""
        self._model.compile(self._optimizer, self._loss, ['accuracy'])

        for iteration in range(self._iterations):
            


def main():
    args = parse()
    model = KerasDense(args)
    model.build()
    model.train()


def parse():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-a',
        '--activation',
        default='relu',
        help='Keras activation'
    )
    parser.add_argument(
        '-l',
        '--loss',
        default='mse',
        help='Keras loss'
    )
    parser.add_argument(
        '-o',
        '--optimizer',
        default='adam',
        help='Keras optimizer'
    )
    parser.add_argument(
        '-e',
        '--extent',
        default=12.,
        type=float,
        help='maximum x and y in the domain'
    )

    parser.add_argument(
        'layers',
        choices=range(1, 5),
        type=int,
        help='number of dense layers'
    )
    parser.add_argument(
        'n_data',
        type=int,
        help='number of data points to train on'
    )
    parser.add_argument(
        'minibatch_size',
        type=int,
        help='minibatch size'
    )

    return parser.parse_args()


# def plot_func():
#     """Plot the underlying function."""
#     x = np.linspace(-12, 12, 1001)
#     x, y = np.meshgrid(x, x)
#     z = f(x, y)
#     fig = plt.gcf()
#     ax = fig.gca(projection='3d')
#     ax.plot_surface(x, y, z, cmap=cm.inferno)
#     plt.show()


if __name__ == '__main__':
    main()
