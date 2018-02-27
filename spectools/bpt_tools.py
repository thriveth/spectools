#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def load_sdssdata():
    sdssdata = Table.read('./bpt_sdss_dr7.txt', format='ascii')
    return sdssdata

def draw_O123_diags(ax):
    ax.plot(np.linspace(-3, 0, 100), -1.70 * np.linspace(-3, 0, 100) - 2.163, 'k-')
    ax.plot(np.linspace(-1.1, 0, 100), np.linspace(-1.1, 0, 100) + .7, 'k-')
    return ax

def draw_N2_O3_diags(ax):
    ax.plot(
        linspace(-4, 0, 200),
        0.61 / (linspace(-4, 0, 200)-0.05) + 1.3,
        'k--', zorder=1, label='Ka03'
    )  # Ka03
    ax.plot(
        linspace(-4, 0.2, 200),
        0.61 / (linspace(-4, 0.2, 200)-0.47) + 1.19,
        'k-', zorder=1, label='Ke01'
    )  # Ke01
    return(ax)

def draw_O13_diags(ax):
    ax.plot(
        linspace(-1.13, -.5, 200),
        1.18 * (linspace(-1.13, -.5, 200)-0.0) + 1.3,
        'k-', zorder=1, label='Ke06')  # Ke06
    ax.plot(
        linspace(-4, -0.89, 200),
        0.73 / (linspace(-4, -0.89, 200) + 0.59) + 1.33,
        'k-', zorder=1) # Ke06
    return ax
