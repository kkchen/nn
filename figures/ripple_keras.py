#!/usr/bin/env python3

"""Train dense networks on a ripple function."""

import argparse
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from tensorflow import keras

from ripple import Ripple

class KerasDense:

    """Class for training Keras dense model."""

    def __init__(self, args):
        self._layers = args.layers
        self._iterations = round(args.n_data / args.minibatch_size)
        self._minibatch_size = args.minibatch_size
        self._activation = args.activation
        self._loss = args.loss
        self._optimizer = args.optimizer
        self._skip = args.skip

        self._model = keras.models.Sequential()
        self._ripple = Ripple(args.extent, self._minibatch_size)

    def build(self):
        """Build the model."""
        n_neurons = (
            (50,), # 80
            (17, 8), # (21, 11),
            (15, 7, 4) # (19, 10, 5)
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
        self._model.compile(self._optimizer, self._loss)

        for iteration in range(self._iterations):
            x, y, z = self._ripple.random_batch()
            loss = self._model.train_on_batch(np.c_[x, y], z)

            if iteration % self._skip == 0:
                print(iteration, loss)

    def save(self):
        """Save the model with the number of layers in the file name."""
        self._model.save('ripple_{}layer.h5'.format(self._layers))

    def save_outputs(self):
        """Save z outputs from the model, with the number of layers."""
        n = 1001
        x = np.linspace(-12, 12, n)
        x, y = np.meshgrid(x, x)
        x = x.flatten()
        y = y.flatten()
        inputs = np.c_[x, y]

        z = self._model.predict(inputs, self._minibatch_size, 1)
        z = np.reshape(z, (n, n))
        np.save('outputs_{}layer'.format(self._layers), z)


def main():
    args = parse()
    model = KerasDense(args)
    model.build()
    model.train()
    model.save()
    model.save_outputs()


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
        '-s',
        '--skip',
        default=128,
        type=int,
        help='interval for displaying loss'
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


if __name__ == '__main__':
    main()
