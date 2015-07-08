
from __future__ import print_function
from __future__ import division

import sys
import time
import os
import argparse
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
        # expected data => [ np.array.2d, ...]
        for i, d in enumerate(data):
            ax = axes[0][i]
            da = data[i]
            im = ax.imshow(da, origin='lower')
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="10%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            del cbar

    elif pltype == 'ploty':
        # expected data => [ [y...], ]
        for i, d in enumerate(data):
            ax = axes[0][i]
            ax.plot(data[i])

    elif pltype == 'plotxy':
        # expected data =>  [ [[x...],[y...]], ]
        for i, d in enumerate(data):
            ax = axes[0][i]
            ax.plot(data[i][0], data[i][1], lw=2)

    else:
        raise ValueError("Unknown pltype => %s" % pltype)

    plt.show()


class DX:
    """ A simple class for reading Open DX files """

    def __init__(self, fname):
        # dx file name
        self.fname = fname

        # Point : the lowermost corner of the grid
        self.corner_lower = None
        self.corner_upper = None
        self.center = None

        # Point : x, y, z spacing
        self.spacing = None

        # Point : x, y, z counts
        self.counts = None

        # np.ndarray : holding the actual data
        self.data = None

        # np.ndarray : holding the data for the current analysis
        self.working_data = None

    def coord_to_index(self, x, y, z):
        if x > self.corner_upper.x or x < self.corner_lower.x:
            raise ValueError('x is not in the range => %f' % x)

        if y > self.corner_upper.y or y < self.corner_lower.y:
            raise ValueError('y is not in the range => %f' % y)

        if z > self.corner_upper.z or z < self.corner_lower.z:
            raise ValueError('y is not in the range => %f' % z)

        ix = (x - self.corner_lower.x) / self.spacing.x
        iy = (y - self.corner_lower.y) / self.spacing.y
        iz = (z - self.corner_lower.z) / self.spacing.z

        return Point(int(ix), int(iy), int(iz))

    def index_to_coord(self, ix, iy, iz):
        if ix >= (self.counts.x):
            raise ValueError('ix is bigger than counts.x => %d' % ix)

        if iy >= (self.counts.y):
            raise ValueError('iy is bigger than counts.x => %d' % iy)

        if iz >= (self.counts.z):
            raise ValueError('iz is bigger than counts.x => %d' % iz)

        x = (ix * self.spacing.x) + self.corner_lower.x
        y = (iy * self.spacing.y) + self.corner_lower.y
        z = (iz * self.spacing.z) + self.corner_lower.z

        return Point(x, y, z)

    def peek(self):
        """ peek will read the initial part of the dx file
            and populate grid properties.
        """

        with open(self.fname) as f:
            tmp_delta = []
            for line in f:
                if line.startswith('#'):
                    continue

                if line.startswith('origin'):
                    xyz = [float(k) for k in line.split()[1:]]
                    self.corner_lower = Point(*xyz)
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

        # for self.corner_upper
        ux = self.corner_lower.x + (self.counts.x) * self.spacing.x
        uy = self.corner_lower.y + (self.counts.y) * self.spacing.y
        uz = self.corner_lower.z + (self.counts.z) * self.spacing.z

        self.corner_upper = Point(ux, uy, uz)

        # for center
        cx = (self.corner_upper.x + self.corner_lower.x) / 2.0
        cy = (self.corner_upper.y + self.corner_lower.y) / 2.0
        cz = (self.corner_upper.z + self.corner_lower.z) / 2.0

        self.center = Point(cx, cy, cz)

    def parse(self, should_cache=False):
        """ parse reads the grid data from the dx file.
            Note to call .peek method first.
        """
        cache_name, ext = os.path.splitext(self.fname)
        cache_name += '.npy'

        if should_cache and os.path.exists(cache_name):
            print('Loading cached data from %s ...' % cache_name)
            self.data = np.load(cache_name)
        else:
            N = self.counts.x * self.counts.y * self.counts.z

            tmp_data = np.zeros(N, dtype='|S12')
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
                            msg = 'Reading %-50s %4d%%' % ('=' * int(m), round(m * 2, 0))
                            print(msg, end='\r')
                            sys.stdout.flush()
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

            if should_cache:
                print('Saving cached data => %s' % cache_name)
                np.save(cache_name, self.data)

        print('Shape => self.data.shape => ', self.data.shape)
        print('Grid is ready.')
        print('')

    def aggregate(self, axis, minv, maxv, action='mean'):
        assert axis in ('x', 'y', 'z')

        print('Averaging for %s axis from %.1f to %.1f' % (axis, minv, maxv))

        if axis == 'x':
            pmin = self.coord_to_index(minv, 0, 0)
            pmax = self.coord_to_index(maxv, 0, 0)
            lo, hi = pmin.x, pmax.x
            ax = 0
            selected = self.data[lo:hi, :, :]
        elif axis == 'y':
            pmin = self.coord_to_index(0, minv, 0)
            pmax = self.coord_to_index(0, maxv, 0)
            lo, hi = pmin.y, pmax.y
            ax = 1
            selected = self.data[:, lo:hi, :]
        elif axis == 'z':
            pmin = self.coord_to_index(0, 0, minv)
            pmax = self.coord_to_index(0, 0, maxv)
            lo, hi = pmin.z, pmax.z
            selected = self.data[:, :, lo:hi]
            ax = 2

        print('Indices => %d %d' % (lo, hi))
        if action == 'mean':
            self.working_data = selected.mean(axis=ax)
        elif action == 'sum':
            self.working_data = selected.sum(axis=ax)
        else:
            raise ValueError('Unknown action => %s' % action)

        print('Shape => self.working_data => ', self.working_data.shape)
        print('')

    def imshow(self):
        """ plots the working data using imshow """

        self.aggregate('z', -70, 70)
        zdata1 = self.working_data

        self.aggregate('z', -30, 30)
        zdata2 = self.working_data

        plotter([zdata1, zdata2], pltype='imshow')

    def analyze(self, maxr=60, limr=40, should_plot=False):
        """
            maxr => the maximum distance from center that should be considered
            limr => the cutoff that defined the distance b/w inside and bulk
            should_plot => will show any intermediate plots
        """

        if limr >= maxr:
            raise ValueError('limr cannot be bigger than maxr')

        print('Analyzing ...')

        rocc = namedtuple('result', 'r occ')
        results = []
        distance = lambda x1, y1, x2, y2: ((x1-x2)**2 + (y1-y2)**2)**0.5

        for ix in range(self.counts.x):
            for iy in range(self.counts.y):
                p = self.index_to_coord(ix, iy, 0)
                r = distance(p.x, p.y, 0.0, 0.0)
                if r > maxr:
                    continue
                results.append(rocc(r, self.working_data[ix, iy]))

        dr = self.spacing.x

        radii = [0 + n * dr for n in range(int(maxr/dr))]
        numbs = [1 for i in range(len(radii) - 1)]

        # values will eventually store g(r)
        values = [0 for i in range(len(radii) - 1)]

        for i, (rmin, rmax) in enumerate(zip(radii[:-1], radii[1:])):
            for rocc in results:
                if rocc.r >= rmin and rocc.r < rmax:
                    numbs[i] += 1
                    values[i] += rocc.occ

        radii = radii[:-1]

        values = np.array(values) / np.array(numbs)

        M = len([x for x in radii if x > limr and x < maxr])
        print('For normalizing, picking the average of last "%d" elements ...' % M)
        values_tail = values[-M:].mean()
        values = values / values_tail

        print('RDF tail is %e => bulk concentration is %f M' % (values_tail, values_tail / (1e-27 * 6.022e23)))

        # plotter([[radii, values]], pltype='plotxy')

        excess = 0
        excess_series = []
        excess_r = []
        adjust_factor = dr / (self.spacing.x * self.spacing.x)

        for i, r in enumerate(radii):
            if r > limr:
                break
            ex = (values[i] - 1) * 2 * r * np.pi * 140 * values_tail * adjust_factor 
            excess += ex
            excess_series.append(excess)
            excess_r.append(r)

        print('Excess ions => %.1f' % excess)
        if should_plot:
            plotter([[radii, values], [excess_r, excess_series]], pltype='plotxy')

        # saving
        #for k in (radii, values, excess_r, excess_series):
        #    n = len(k)
        #    k = np.array(k).reshape(n, 1)
        np.savetxt('rdf_' + self.fname ,  np.array([radii, values]).T)
        np.savetxt('excess_' + self.fname,  np.array([excess_r, excess_series]).T)

    def __repr__(self):
        info = '\nOpenDX file with the following properties:\n'
        info += '  lower   => %6.2f %6.2f %6.2f\n' % self.corner_lower
        info += '  center  => %6.2f %6.2f %6.2f\n' % self.center
        info += '  upper   => %6.2f %6.2f %6.2f\n' % self.corner_upper
        info += '  counts  => %6d %6d %6d\n' % self.counts
        info += '  spacing => %6.2f %6.2f %6.2f\n' % self.spacing
        return info


def main():
    p = argparse.ArgumentParser()

    p.add_argument('--cmd', default='info', required=True,
                   help='command to be used: info, analyze, test')

    p.add_argument('--dx', required=True,
                   help='name of the dx file')

    p.add_argument('--minz', default=-30, type=float,
                   help='min z')

    p.add_argument('--maxz', default=+30, type=float,
                   help='max z')

    p.add_argument('--maxr', default=+60, type=float,
                   help='max r')

    p.add_argument('--limr', default=+40, type=float,
                   help='inside-bulk cutoff')

    p.add_argument('--cache', action='store_true', default=False,
                   help='cache processed data')

    p.add_argument('--plot', action='store_true', default=False,
                   help='show plots')

    args = p.parse_args()

    dx = DX(args.dx)
    dx.peek()
    print(dx)

    if args.cmd == 'info':
        pass

    elif args.cmd == 'analyze':
        dx.parse(should_cache=args.cache)
        if args.plot:
            dx.imshow()
        dx.aggregate('z', args.minz, args.maxz)
        dx.analyze(maxr=args.maxr, limr=args.limr, should_plot=args.plot)

    if sys.argv[1] == 'test':
        print(dx.coord_to_index(0, 0, 0))
        print(dx.index_to_coord(129, 149, 154))


if __name__ == '__main__':
    main()

