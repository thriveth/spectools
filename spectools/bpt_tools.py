#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('spectools', 'data/')
bpt_data = pkg_resources.resource_filename('spectools', 'data/bpt_sdss_dr7.txt')


def load_sdssdata():
    sdssdata = Table.read(bpt_data, format='ascii')
    return sdssdata

def plot_bpt(fluxes=None, diagnostic='OIII', ax=None):
    'Diagnostic can be OI or OIII'
    if not ax:
        fig, ax = plt.subplots(1)
    clouddata = load_sdssdata()
    xcloud = 'OI/Ha' if diagnostic=='OI' else 'NII/Ha'
    ycloud = 'OIII/Hb'
    sdsscloud = ax.hist2d(
        clouddata['log '+xcloud],
        clouddata['log '+ycloud],
        cmap='Greys',
        bins=50,
        norm=mpl.colors.LogNorm(),
        range=[[-2.5, 0.5], [-1.5, 1.5]],
        vmax=100000,
    )
    if diagnostic == 'OIII':
        ax = draw_N2_O3_diags(ax)
    elif diagnostic == 'OI':
        ax = draw_O13_diags(ax)
    return ax

def abund_sequence(z):
    """ Draws the Star formation abundance tracks of Kewley+ 2013 (eq. 5) as a
    function of redshift.
    """
    # TODO: Needs to fix the limits of interval of definition somehow, it
    # changes with redshift and plots ugly artifacts when not correct.
    xs = np.linspace(-2, 0.3, 1000)
    denom = xs + 0.08 - 0.1833 * z
    ys = 1.1 + 0.03 * z + 0.61/denom
    return xs, ys

def draw_O123_diags(ax):
    ax.plot(np.linspace(-3, 0, 100), -1.70 * np.linspace(-3, 0, 100) - 2.163, 'k-')
    ax.plot(np.linspace(-1.1, 0, 100), np.linspace(-1.1, 0, 100) + .7, 'k-')
    return ax

def draw_N2_O3_diags(ax):
    ax.plot(
        np.linspace(-4, 0, 200),
        0.61 / (np.linspace(-4, 0, 200)-0.05) + 1.3,
        'k--', zorder=1, label='Ka03'
    )  # Ka03
    ax.plot(
        np.linspace(-4, 0.2, 200),
        0.61 / (np.linspace(-4, 0.2, 200)-0.47) + 1.19,
        'k-', zorder=1, label='Ke01'
    )  # Ke01
    return ax

def draw_O13_diags(ax):
    ax.plot(
        np.linspace(-1.13, -.5, 200),
        1.18 * (np.linspace(-1.13, -.5, 200)-0.0) + 1.3,
        'k-', zorder=1, label='Ke06')  # Ke06
    ax.plot(
        np.linspace(-4, -0.89, 200),
        0.73 / (np.linspace(-4, -0.89, 200) + 0.59) + 1.33,
        'k-', zorder=1) # Ke06
    return ax
