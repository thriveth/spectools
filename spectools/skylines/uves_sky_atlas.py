#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Insert module docstring
"""

import numpy as np
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import pkg_resources


DATA_PATH = pkg_resources.resource_filename('spectools', 'data/')
sky_data = pkg_resources.resource_filename('spectools', 'data/UVES_sky_tables/')

_wave_coverage_dict = {
    '346':  [3142.0, 3800.0],
    '437':  [3747.0, 4848.0],
    '580L': [4861.0, 5697.0],
    '580U': [5865.0, 6718.0],
    '800U': [8541.0, 8637.0],
    '860L': [6704.0, 8539.0],
    '860U': [8614.0, 10419.0]
}

_wavcovframe = pd.DataFrame.from_dict(_wave_coverage_dict).T
_wavcovframe.sort_values(0, inplace=True)
_linefiles = ['346', '437', '580L', '580U', '800U', '860L', '860U']

class UvesSkyAtlas(object):
    """ Init docstring goes here

    Data wavelengths are all air (right? That's what they are, right?)
    """
    def __init__(self, wavelims=None):
        if wavelims is None:
            wavelims = [3000, 11000]
        self._wavelims = wavelims
        self.ax = None
        self.plotlines = None
        self.line_collections = {}
        self.data_collections = {}
        self.load_lines()

    def load_lines(self):
        """ Loads the lines
        """
        # min_file = np.absolute(  # TODO Make this better, somehow.
        #     _wavcovframe[0] - min(self._wavelims)
        # ).argmin()
        # max_file = np.absolute(_wavcovframe[1] - max(self._wavelims)).argmin()
        partframe = self.partframe
        for a in partframe.index:
            t = Table.read(
                sky_data+'gident_{}.dat'.format(a),
                # './skylines/UVES_sky_tables/gident_{}.dat'.format(a),
                format='ascii.fixed_width_two_line'
            )
            self.line_collections[a] = t

    def load_data(self):
        """ Docstring here
        """
        # TODO be a bit smart about it, check whether arrays are already loaded,
        # delete unneeded ones, only load missing ones.
        raise NotImplementedError(
            "Loading actual data not implemented yet \n\
            So far, make do with line centroids and strengths")

    @property
    def wavelims(self):
        """ Get the wavelength limits """
        return self._wavelims

    @wavelims.setter
    def wavelims(self, wavlim):
        try:
            # Testing whether input has correct format
            for i in wavlim: pass
            a = i * i
        except TypeError:
            raise
        if len(wavlim) == 2:
            self._wavelims = wavlim
        self.load_lines()

    @property
    def partframe(self):
        min_file = np.absolute(
            _wavcovframe[0] - min(self.wavelims)
        ).argmin()
        max_file = np.absolute(_wavcovframe[1] - max(self.wavelims)).argmin()
        partframe = _wavcovframe[min_file:max_file]
        return partframe
