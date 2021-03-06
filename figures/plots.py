#!/usr/bin/env python3

"""Template for making figure plots."""

import argparse
from multiprocessing import Pool
import os
import sys

from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ripple import Ripple

class Plots:

    """Class for making figure plots."""

    _fontsize = 9

    def __init__(self):
        """Parse the command line argument(s) and save the tag."""
        self._parse()

        plt.rc('text', usetex=True)
        plt.rc('font', size=self._fontsize, family='serif')

    def plot(self):
        """Run the plotting method based on the tag; save or display."""
        if 'cluster' in self.tag:
            self._set_cluster_data()

        if self.nprocs == 1:
            # Have this option available in case Pool doesn’t behave.
            for t in self.tag:
                self._run_tag(t)
        else:
            with Pool(self.nprocs) as pool:
                pool.map(self._run_tag, self.tag)

        if self.verbose:
            print('Finished.')

    def _plot_linear_regression(self):
        """Plot a linear regression example."""
        n = 50
        x = np.random.uniform(0, 1, n)
        y = x + np.random.normal(0, 0.2, n)
        z = np.arange(0, 2)
        p = np.polyfit(x, y, 1)

        plt.figure(figsize=(1.75, 1.1))
        ax = plt.axes((0, 0, 1, 1))
        ax.axis('off')
        ax.plot(x, y, '.', ms=3)
        ax.plot(z, np.polyval(p, z), 'r-')

    def _plot_cluster(self):
        """Plot a monochrome example of a cluster."""
        self._cluster_plot(False)

    def _plot_clusters(self):
        """Plot a colored example of clusters."""
        self._cluster_plot(True)

    def _plot_sigmoid(self, figsize=(1.8, 1.2)):
        """Plot the standard logistic function."""
        x = np.linspace(-5, 5, 10001)
        y = 1 / (np.exp(-x) + 1)

        plt.figure(figsize=figsize)
        plt.plot(x, y, 'g')
        plt.grid(True)
        plt.axis((-5, 5, 0, 1))
        plt.xlabel('$y$')
        plt.ylabel('$\sigma$')
        plt.gca().set_yticks(np.arange(0, 1.5, 0.5))
        plt.tight_layout(0)

    def _plot_plane(self):
        """Plot a hyperplane example."""
        # Color, from blue (s = 0) to red (s = 1).
        color = lambda s: (s, 0, 1 - s)

        plt.figure(figsize=(2.7, 1.4))

        for i in range(1000):
            x = np.random.uniform()
            y = np.random.uniform()
            s = min(max(0, (x + y) - 0.5 + np.random.normal(scale=0.5)), 1)

            plt.plot(x, y, '.', color=color(s), ms=1)

        plt.plot((0, 1), (1, 0), 'g-', lw=2)

        plt.axis((0, 1, 0, 1))
        plt.xlabel('age')
        plt.ylabel('LDL')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.tight_layout(0)

    def _plot_activations(self):
        """Plot common activation functions."""
        self._plot_sigmoid((2, 1.6))
        x = np.linspace(-5, 5, 10001)
        zeros = np.zeros(x.shape)
        relu = np.maximum(zeros, x)

        plt.plot(x, relu + np.minimum(zeros, np.exp(x) - 1), 'm')
        plt.plot(x, relu + np.minimum(zeros, 0.05 * x), 'r')
        plt.plot(x, relu, 'b')

        plt.axis((-4, 4, -1, 4))
        plt.xlabel('$x$')
        plt.gca().set_xticks(range(-4, 6, 2))
        plt.gca().set_yticks(range(-1, 5))
        plt.tight_layout(0)

    def _plot_ripple(self):
        """The ripple ground truth."""
        ax, x, y = self._pre_ripple_plot()
        ax.plot_surface(x, y, Ripple.f(x, y), cmap=cm.inferno)

    def _plot_ripple_1layer(self):
        """The ripple with 1 hidden layer."""
        self._n_layer_ripple(1)

    def _plot_ripple_2layer(self):
        """The ripple with 2 hidden layer."""
        self._n_layer_ripple(2)

    def _plot_ripple_3layer(self):
        """The ripple with 3 hidden layer."""
        self._n_layer_ripple(3)

    def _plot_minima(self):
        """Plot a surface with multiple local minima."""
        x = np.linspace(-1, 1, 1001)
        y = np.linspace(-1, 1, 1001)
        x, y = np.meshgrid(x, y)

        z = (
            -2 * np.exp(-(16 * ((x - 0.2) ** 2 + (y - 0.4) ** 2)))
            - 1.3 * np.exp(-(8 * ((x + 0.4) ** 2 + (y + 0.1) ** 2)))
            - 0.8 * np.exp(-(10 * ((x - 0.3) ** 2 + (y + 0.6) ** 2)))
        )
        fig = plt.figure(figsize=(3.5, 2.3))
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z, cmap=cm.jet_r)
        ax.axis('off')
        ax.dist = 8

    def _plot_under_over_train(self):
        """Plot under/properly/over-trained model."""
        x = np.random.uniform(0, 2 * np.pi, 100)
        x = np.sort(x)
        y = np.sin(x) + np.random.normal(0, 0.4, len(x))

        plt.figure(figsize=(4.25, 1.5))
        ax = [plt.axes((0.045 + i * 0.32, 0.11, 0.31, 0.75)) for i in range(3)]

        z = np.array((0, 1, 4, 2 * np.pi))
        ax[0].plot(z, np.sin(z))

        z = np.linspace(0, 2 * np.pi, 20)
        ax[1].plot(z, np.sin(z))

        ax[2].plot(x, y)

        for a in ax:
            a.plot(x, y, '.', ms=1)

            a.set_xlabel('$x$')
            a.set_xticks([])
            a.set_yticks([])

        ax[0].set_ylabel('$y$')
        ax[0].set_title('Underfitted')
        ax[1].set_title('Good model')
        ax[2].set_title('Overfitted')

    def _plot_extended_ripple(self):
        """Plot the ripple on a larger domain."""
        ax, x, y = self._pre_extrapolate_ripple()
        ax.plot_surface(x, y, Ripple.f(x, y), cmap=cm.inferno)

    def _plot_extrapolated_ripple(self):
        z = np.load('extrapolate_2layer.npy')
        ax, x, y = Plots._pre_extrapolate_ripple()
        ax.plot_surface(x, y, z, cmap=cm.inferno)

    @staticmethod
    def _plot_loss():
        """Plot example training and test loss."""
        plt.figure(figsize=(4.25, 1.1))

        x = np.arange(10000)
        train_loss = (0.07 * x + 1) ** -0.3 + np.random.normal(0, 0.01, len(x))
        test_loss_noiseless = (
            (0.07 * x + 1) ** -0.3
            + 0.04
            + 0.05 * np.log(np.exp((x - 7000) / 1000) + 1)
        )
        argmin = np.argmin(test_loss_noiseless)
        test_loss = test_loss_noiseless + np.random.normal(0, 0.01, len(x))

        plt.plot(x, train_loss, 'b', x, test_loss, 'g', lw=0.5)
        plt.plot(argmin, test_loss_noiseless[argmin], 'ro')
        plt.grid(True)
        plt.axis((0, 10000, 0, 1))
        plt.xlabel('iteration')
        plt.ylabel('$L$')
        plt.legend(('train', 'test'), loc='upper right', ncol=2)
        plt.tight_layout(0)

    def _set_cluster_data(self):
        """Get the 2-D cluster data."""
        n = 100

        self._cluster_x = [
            np.random.normal(-1, 1, n),
            np.random.normal(0, 0.7, n),
            np.random.normal(2, 1.2, n)
        ]
        self._cluster_y = [
            np.random.normal(3, 1, n),
            np.random.normal(-1, 0.7, n),
            np.random.normal(2, 1.2, n)
        ]

    def _cluster_plot(self, colors):
        """Make a plot of the clusters, signaling whether to use colors."""
        plt.figure(figsize=(1.3, 1.3))
        ax = plt.axes((0, 0, 1, 1))
        fmt = '.'

        if not colors:
            fmt += 'k'

        for a, b in zip(self._cluster_x, self._cluster_y):
            ax.plot(a, b, fmt, ms=1)

        ax.axis('off')

    def _parse(self):
        """Parse the command-line arguments."""
        prefix = '_plot_'
        methods = [m[len(prefix):]
                   for m in dir(self) if m[:len(prefix)] == prefix]
        methods.insert(0, 'all')

        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument(
            '-n',
            '--nprocs',
            default=os.cpu_count(),
            type=int,
            help='number of processors to use'
        )
        parser.add_argument(
            '-s',
            '--save',
            action='store_true',
            help='save the figure to disk'
        )
        parser.add_argument(
            '-d',
            '--display',
            action='store_true',
            help='display the plot'
        )
        parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help='verbose output'
        )

        parser.add_argument(
            'tag',
            nargs='*',
            default='all',
            choices=methods,
            help='name of plot to produce; if not provided, plot all'
        )

        parser.parse_args(namespace=self)

        if 'all' in self.tag:
            self.tag = methods[1:]

        if not (self.save or self.display):
            raise RuntimeError('At least one of -s and -d must be given.')

    def _run_tag(self, tag):
        """Run the given plotting tag."""
        if self.verbose:
            print('Plotting {}.'.format(tag))

        getattr(self, '_plot_' + tag)()
        use_png = 'ripple' in tag or tag == 'minima'

        if self.save:
            filename = '{}.{}'.format(tag, 'png' if use_png else 'eps')

            if self.verbose:
                print('Saving {}.'.format(filename))

            if use_png:
                plt.savefig(filename, dpi=300)
            else:
                plt.savefig(filename)

        if self.display:
            if self.verbose:
                print('Displaying {}.'.format(tag))

            plt.show()

    @staticmethod
    def _label(ax, label, xshift, yshift):
        """Place a subfigure label on the upper left of the axes."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if ax.get_xscale() == 'linear':
            x = xlim[0] - (xlim[1] - xlim[0]) * xshift
        else:
            x = xlim[0] / (xlim[1] / xlim[0]) ** xshift

        if ax.get_yscale() == 'linear':
            y = ylim[1] - yshift * (ylim[1] - ylim[0])
        else:
            y = ylim[1] / (ylim[1] / ylim[0]) ** yshift

        plt.text(x, y, '({})'.format(label))

    @staticmethod
    def _pre_ripple_plot():
        """Commands to run before making a ripple plot."""
        x = np.linspace(-12, 12, 1001)
        x, y = np.meshgrid(x, x)

        fig = plt.figure(figsize=(2.1, 1.6))
        ax = fig.gca(projection='3d')
        plt.axis('off')
        ax.dist = 6.75

        return ax, x, y

    @staticmethod
    def _n_layer_ripple(n):
        z = np.load('outputs_{}layer.npy'.format(n))
        ax, x, y = Plots._pre_ripple_plot()
        ax.plot_surface(x, y, z, cmap=cm.inferno)

    @staticmethod
    def _pre_extrapolate_ripple():
        """Commands to run before making big (extrapolated) ripple plots."""
        x = np.linspace(-18, 18, 1501)
        x, y = np.meshgrid(x, x)

        fig = plt.figure(figsize=(3.15, 2.4))
        ax = fig.gca(projection='3d')
        plt.axis('off')
        ax.dist = 6.75

        return ax, x, y

def main():
    plots = Plots()
    plots.plot()


if __name__ == '__main__':
    main()
