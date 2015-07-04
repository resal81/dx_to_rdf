
import sys
import time
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# used to hold simple coordinate data
Point = namedtuple('Point', 'x y z')


def plotter(data=[], pltype=None):
    N = len(data)
    fig, axes = plt.subplots(1, N, squeeze=False)

    if pltype == 'imshow':
        for i, d in enumerate(data):
            ax = axes[0][i]
            im = ax.imshow(data[i], origin='lower')
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="20%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            del cbar
        # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        # fig.colorbar(im, cax=cax)
    else:
        raise ValueError("Unknown pltype => %s" % pltype)

    plt.show()


class DX:
    """ A simple class for reading Open DX files """

    def __init__(self, fname):
        self.fname = fname
        self.origin = None
        self.spacing = None
        self.counts = None
        self.data = None

    def peek(self):
        with open(self.fname) as f:
            tmp_delta = []
            for line in f:
                if line.startswith('#'):
                    continue

                if line.startswith('origin'):
                    xyz = [float(k) for k in line.split()[1:]]
                    self.origin = Point(*xyz)
                    continue

                if line.startswith('object'):
                    if line.startswith('object 1 class gridpositions'):
                        xyz = [int(k) for k in line.split()[-3:]]
                        self.counts = Point(*xyz)
                    continue

                if line.startswith('delta'):
                    tmp_delta.append(line.split()[1:])
                    continue
                break

        dx = None
        dy = None
        dz = None

        zero_str = '0'
        for delta in tmp_delta:
            if delta[1] == delta[2] == zero_str:
                dx = float(delta[0])
            elif delta[0] == delta[2] == zero_str:
                dy = float(delta[1])
            elif delta[0] == delta[1] == zero_str:
                dz = float(delta[2])
            else:
                print('tmp_delta => ', tmp_delta)
                raise ValueError("could not parse delta block")

        self.spacing = Point(dx, dy, dz)

    def parse(self):
        N = self.counts.x * self.counts.y * self.counts.z

        tmp_data = np.zeros(N, dtype='|S4')
        self.data = np.zeros((self.counts.x, self.counts.y, self.counts.z))

        counter = 0
        t1 = time.time()
        with open(self.fname) as f:
            for line in f:
                if line.strip() == '':
                    continue
                if line.startswith(('#', 'object', 'origin', 'delta')):
                    continue

                points = line.split()

                for p in points:
                    tmp_data[counter] = p

                    counter += 1
                    if counter % 50000 == 0:
                        m = (counter / N) / 0.02
                        msg = 'Reading (%-50s) %4.1f%%' % ('=' * int(m), m * 2)
                        print(msg, end='\r')
        print('')
        t2 = time.time()
        print('Finished reading file in %.1f seconds' % (t2-t1))

        print('Converting to float ...')
        tmp_data = tmp_data.astype(np.float)

        print('Setting up the grid ...')
        counter = 0
        for ix in range(self.counts.x):
            for iy in range(self.counts.y):
                for iz in range(self.counts.z):
                    self.data[ix, iy, iz] = tmp_data[counter]
                    counter += 1

        print('Grid is ready.')

    def analyze(self):
        print('Analyzing ...')
        print('Shape => self.data.shape => ', self.data.shape)

        av_z = self.data.mean(axis=2)
        print('Shape => av_z.shape => ', av_z.shape)

        plotter([av_z], pltype='imshow')

    def __repr__(self):
        info = 'Open DX file with:'
        info += '  origin => %.2f %.2f %.2f' % self.origin
        info += '  counts => %d %d %d' % self.counts
        info += '  spacing=> %.2f %.2f %.2f' % self.spacing
        return info


def main():
    if len(sys.argv) <= 2:
        usage = '\nUsage:\n'
        usage += '   python3 %s [cmd] dxfile\n' % __file__
        usage += '\nCommands: peek, \n'
        print(usage)
        return

    if sys.argv[1] == 'peek':
        dx = DX(sys.argv[2])
        dx.peek()
        print(dx)
        dx.parse()
        dx.analyze()

if __name__ == '__main__':
    main()

