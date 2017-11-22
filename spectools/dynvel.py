#! /usr/bin/env/ python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
from astropy.table import Table
from astropy.io import ascii
import astropy.units as u
from spectools.spectools import SimpleMaskGUI

class DynVelGUI(SimpleMaskGUI):
    """ Computes dynamic velocity properties of an absorption line"""
    plot_added = None
    iterations = 100
    def _onselect(self, event):
        pass
    def _on_clr_button(self, event):
        pass
    def _on_ok_button(self, event):
        pass


def dynvel(wave, data, errs, iterations=100, kind='absorb'):
    """Computes a range of kinematic properties from a spectral line.
    The line is assumed continuum normalized.
    EW is given in km/s, as the function only knows velocity and flux.
    """
    # Create `iterations` * len(data) arrays, repeating data, errs, wave
    # `iterations` times. Basically stacking the arrays vertically `iterations`
    # times.
    # TODO Possibly interpolate on finer vel grid to get more precise numbers?
    diff = np.diff(wave)
    diff = np.append(diff, diff[-1])
    diffs = diff.repeat(iterations).reshape(-1, iterations).T
    waves = wave.repeat(iterations).reshape(-1, iterations).T
    datas = data.repeat(iterations).reshape(-1, iterations).T - 1
    errss = errs.repeat(iterations).reshape(-1, iterations).T
    # Generate perturbation array, first row is unperturbed
    perturb = np.random.normal(loc=0, scale=np.absolute(errss))
    perturb[0, :] = 0
    pertdata = datas + perturb
    # Flux is positive, even if it is absorbed flux. Dixi!
    if kind == 'absorb':
        pertdata *= -1
    # Accumulated flux, first row still original data.
    accu = np.cumsum(pertdata*diffs, axis=1)
    EWs = (pertdata * diffs).sum(axis=1)
    for i in np.arange(iterations):
        accu[i, :] /= (pertdata*diffs).sum(1)[i]
    five = np.absolute(accu - 0.05).argmin(axis=1)
    fifty = np.absolute(accu - 0.5).argmin(axis=1)
    ninetyfive = np.absolute(accu - 0.95).argmin(axis=1)
    mins = pertdata.argmax(axis=1)
    fivevels = np.array(
        [waves[i, five[i]] for i in np.arange(waves.shape[0])]
    )
    fiftyvels = np.array(
        [waves[i, fifty[i]] for i in np.arange(waves.shape[0])]
    )
    ninetyfivevels = np.array(
        [waves[i, ninetyfive[i]] for i in np.arange(waves.shape[0])]
    )
    w90vels = np.array(
        [waves[i, ninetyfive[i]] - waves[i, five[i]]
         for i in np.arange(waves.shape[0])]
    )
    minvels = np.array(
        [waves[i, mins[i]] for i in np.arange(waves.shape[0])]
    )
    # Now, find the various characteristic velocities
    minvel = {
        'ml':minvels[0],
        'mean':minvels[1:].mean(),
        'stddev':minvels[1:].std()
    }
    fivevel = {
        'ml': fivevels[0],
        'mean': fivevels[1:].mean(),
        'stddev': fivevels[1:].std(),
        'percentiles': np.percentile(fivevels[1:], [2.5, 16, 50, 84, 97.5]),
        'realizations': fivevels,
    }
    fiftyvel = {
        'ml': fiftyvels[0],
        'mean': fiftyvels[1:].mean(),
        'stddev':fiftyvels[1:].std()
    }
    ninetyfivevel = {
        'ml': ninetyfivevels[0],
        'mean': ninetyfivevels.mean(),
        'stddev': ninetyfivevels.std()
    }
    w90vel = {
        'ml':  w90vels[0],
        'mean': w90vels.mean(),
        'stddev': w90vels.std(),
    }
    EW = {
        'ml': EWs[0],
        'mean': EWs[1:].mean(),
        'percentiles': np.percentile(EWs[1:], [2.5, 16, 50, 84, 97.5]),
        'realizations': EWs,
    }
    outdict = {
        'EW': EW,
        'iterations':iterations,
        'vmin': minvel,
        'v5pct': fivevel,
        'vint': fiftyvel,
        'v95': ninetyfivevel,
        'w90': w90vel,
        'accu': accu[0, :],
        'perturbed': pertdata,
        'cumsum': accu,
        'Velocity': wave,
    }
    return outdict


def dynvelplot(indict, ax=None, plotiters=True, color1='black', color2='C0'):
    """Takes as input the output dict from the dynvel() function"""
    if not ax:
        fig, ax = plt.subplots(1)
    ax2 = plt.twinx(ax=ax)
    vels = indict['Velocity']
    if plotiters:
        iterations = max(indict['iterations'], 500)
        for i in np.arange(iterations):
            ax2.plot(vels, indict['cumsum'][i, :], color=color1, ls='-', lw=0.1, alpha=.2)
            #ax.plot(vels, 1-indict['perturbed'][i, :], color=color2, ls='-', lw=0.1, alpha=.2)
    accuplot = ax2.plot(vels, indict['cumsum'][0, :], '-', color='m', lw=1.8, label='Accumulated absorption', )
    fluxplot = ax.plot(vels, 1-indict['perturbed'][0, :], 'c-', lw=1.8, label='Line flux', drawstyle='steps-mid')
    flerrs = ax.fill_between(
        vels,
        1 - indict['perturbed'][0, :] - indict['perturbed'].std(0),
        1 - indict['perturbed'][0, :] + indict['perturbed'].std(0),
        color='C0',
        alpha=.5,
        zorder=0,
        step='mid'
    )
    ax.axvline(0, color='k', ls=':')
    ax.axhline(1, color='k', ls=':')
    ax.axhline(0, color='k', ls='--')
    ax.axvline(indict['v5pct']['ml'], color='darkgray', ls='--')
    #ax.axvspan(
    #    indict['v5pct']['ml']-indict['v5pct']['stddev'],
    #    indict['v5pct']['ml'] + indict['v5pct']['stddev'],
    #    facecolor='lightgray', edgecolor='darkgray', alpha=.5, zorder=0,
    #)
    ax.axvline(
        indict['v95']['ml'],
        indict['v95']['mean'],
        color='darkgray', ls='--'
    )
    #ax.axvspan(
    #    indict['v95']['ml']-indict['v95']['stddev'],
    #    indict['v95']['ml'] + indict['v95']['stddev'],
    #    facecolor='lightgray', edgecolor='darkgray',
    #    alpha=.5, zorder=0,
    #)
    ax.axvline(
        indict['vint']['ml'],
        indict['vint']['mean'],
        color='darkgray',
        ls='--'
    )
    # ax.axvspan(
    #     indict['vint']['ml'] - indict['vint']['stddev'],
    #     indict['vint']['ml'] + indict['vint']['stddev'],
    #     facecolor='lightgray',
    #     edgecolor='darkgray',
    #     alpha=.5, zorder=0,
    # )
    ax.axvline(
        indict['vmin']['ml'],
        indict['vmin']['mean'],
        color='darkgray', ls='--'
    )
    # ax.axvspan(
    #     indict['vmin']['ml']-indict['vmin']['stddev'],
    #     indict['vmin']['ml'] + indict['vmin']['stddev'],
    #     facecolor='lightgray', edgecolor='darkgray',
    #     alpha=.5, zorder=0,
    # )

    ax2.set_ylim(ax.get_ylim())
    ax.tick_params(axis='y', labelcolor=color2)
    ax2.tick_params(labelcolor=color1)
    ax.set_ylabel("Normalized flux", color=color2)
    ax2.set_ylabel("Normalized accumulated absorption", color=color1)
    ax.set_xlabel(r'$v - v_0$ [km/s]')
    lns = accuplot + fluxplot
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='upper left', frameon=False)
    return fig, ax, ax2


def pprint_dynvel(indict):
    s1 = ''.join([r"$v_{5\%} =", "{:.0f} \pm {:.0f}$ km/s".format(indict['v5pct']['ml'], indict['v5pct']['stddev']),])
    s2 = ''.join([r"$v_{95\%} =", "{:.0f} \pm {:.0f}$ km/s".format(indict['v95']['ml'], indict['v95']['stddev']),])
    s3 = ''.join([r"$v_{\mathrm{int}} =", "{:.0f} \pm {:.0f}$ km/s".format(indict['vint']['ml'], indict['vint']['stddev']),])
    s4 = ''.join([r"$v_{min} =", "{:.0f} \pm {:.0f}$ km/s".format(indict['vmin']['ml'], indict['vmin']['stddev']),])
    return "\n".join([s1, s3, s4, s2])

