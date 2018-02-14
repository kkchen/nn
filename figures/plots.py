#!/usr/bin/env python3

"""Template for making figure plots."""

import argparse
from multiprocessing import Pool
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np

class Plots:

    """Class for making figure plots."""

    _fontsize = 10

    def __init__(self):
        """Parse the command line argument(s) and save the tag."""
        self._parse()

        rc('text', usetex=True)
        rc('font', size=self._fontsize)

    def plot(self):
        """Run the plotting method based on the tag; save or display."""
        if 'cluster' in self.tag or 'clusters' in self.tag:
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
        n = 100
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
        self._plot_sigmoid((2, 2.2))
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

        if self.save:
            if self.verbose:
                print('Saving {}.eps.'.format(tag))

            plt.savefig(tag + '.eps')

        if self.display:
            if self.verbose:
                print('Displaying {}.'.format(tag))

            plt.show()

    def _label(self, ax, label, xshift, yshift):
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


def main():
    plots = Plots()
    plots.plot()


if __name__ == '__main__':
    main()
