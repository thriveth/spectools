#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .spectools import GalaxySpectrum, SimpleFitGUI, SpecView
from .helper_functions import wl_to_v, v_to_wl, v_to_deltawl, deltawl_to_v
from matplotlib.widgets import CheckButtons, SpanSelector, RadioButtons, Button
from astropy.io.misc import yaml
from astropy.table import Table
from astropy.convolution import interpolate_replace_nans, convolve
from statsmodels.nonparametric.kernel_regression import KernelReg

class LyaGUI(SimpleFitGUI):
    """ This is a GUI to perform model-independent measurements of Ly-alpha line
    profiles (primarily interesting for emission).

    It measures the following craracteristics, when present:
        - Equivalent Width
        - Red and blue peak velocity and separation
        - Red and blue peak flux relative to continuum, and their ratio
        - Red and blue integrated flux, and their ratio
        - Peak-to-valley flux ratios.
        - Asymmetry (crookedity?) of red peak
    """
    galaxy_name = ''
    data = None
    interp = None
    errors = None
    # mask = None
    smoothed = None
    num_realizations = 1000
    z = 0.
    summary_dict = {}

    _current_peak = 'Red'
    _peaks = np.array(['Red', 'Blue', 'Valley'])
    _peak_on = np.array([True, True, True])
    _colors = {'Red': 'tab:red', 'Blue': 'tab:blue', 'Valley': '0.7'}
    _extra_plots = {}

    def __init__(self, summaryfile=None, inwave=None, indata=None, inerrs=None,
                 inmask=None, smooth=None):
        self.data = indata
        self.wave = inwave
        self.errs = inerrs
        self.mask = inmask
        if summaryfile:
            self.open_summary(summaryfile)
            # Interpolate masked areas
            self.data[self.mask] = np.nan
            self.nonnanidx = np.where(~self.mask)[0]
            self.interp = np.interp(
                self.wave, self.wave[self.nonnanidx], self.data[self.nonnanidx])
            self.interr = np.interp(
                self.wave, self.wave[self.nonnanidx], self.errs[self.nonnanidx])
        if smooth == 'll':
            lle = KernelReg(self.interp, self.wave, 'c', bw=[10])
            mean, marg = lle.fit()
            del marg
            self.smoothed = mean
        elif smooth == 'box':
            mean = np.convolve(self.data, np.array([1, 1, 1])/3)
        else:
            self.smoothed = self.data
        self._build_plot()

    def open_summary(self, summary_file):
        # TODO Fix summaries to contain errors. Unfortunately, that means
        # redoing the whole darn thing.... Sigh.
        with open(summary_file) as f:
            summary = yaml.load(f)
        self.z = summary['redshift']
        self.galaxy_name = summary['galaxyname']
        self.data = np.array(summary['transitions']['Ly alpha']['data'])
        self.errs = np.array(summary['transitions']['Ly alpha']['errs'])
        self.wave = np.array(summary['transitions']['Ly alpha']['wave'])
        self.mask = np.array(summary['transitions']['Ly alpha']['mask'])
        self._cfp = summary['transitions']['Ly alpha']['cont_fit_params']
        self._cont = self.wave * self._cfp['slope'] + self._cfp['intercept']


    def _build_plot(self):
        self.fig = plt.figure(figsize=(8, 4))
        self.ax = self.fig.add_axes([0.08, 0.08, 0.8, 0.8])
        self.chax = self.fig.add_axes([0.90, 0.55, 0.09, 0.3], frameon=False)
        self.chax.set_title('Components \npresent', size='x-small')
        self.rax = self.fig.add_axes([0.90, 0.25, 0.09, 0.2], frameon=False)
        self.rax.set_title('Current \ncomponent', size='x-small')
        self.okax = self.fig.add_axes([0.9, 0.05, 0.09, 0.08])
        self.ok_button = Button(self.okax, 'Done')
        self.ok_button.on_clicked(self._ok_clicked)
        self.reax = self.fig.add_axes([0.9, 0.15, 0.09, 0.08])
        self.re_button = Button(self.reax, 'Reset')
        self.re_button.on_clicked(self._reset_clicked)
        for ax in self.chax, self.rax:
            ax.tick_params(length=0, labelleft='off', labelbottom='off')
        self.ax.plot(self.wave, self.data, 'k-', drawstyle='steps-mid')
        if self.interp is not None:
            self.ax.plot(self.wave, self.interp, '-', color='C1', zorder=0)
        # if self.smoothed is not None:
        #     self.ax.plot(self.wave, self.smoothed, '-', color='C2')
        self.ax.axhline(0, ls='-', color='k', lw=1)
        self.ax.axvline(1215.67 * (1 + self.z), ls=':', color='k', lw=.8)
        self.ax.axhline(1, ls=':', color='k', lw=.8)
        # Insert check buttons
        clabels = self._peaks
        cons = self._peak_on
        self.check = CheckButtons(self.chax, clabels, cons)
        self.check.on_clicked(self._check_clicked)
        rlabels = clabels[np.where(cons)]
        self.radio = RadioButtons(self.rax, rlabels)
        # self.radio.on_clicked(self._radio_clicked)
        self._selector = SpanSelector(
            self.ax, self._onselect, 'horizontal',
            minspan=self._wave_diffs.mean() * 5, )
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel('Normalized flux')

    def _onselect(self, xmin, xmax):
        self.fit_active(xmin, xmax)


    def _ok_clicked(self, event):
        self.save_summary()
        plt.close(self.ax.figure)

    def _reset_clicked(self, event):
        self.summary_dict = {}
        for a in self._extra_plots.values(): a.remove()
        print('You presset RESET and are now back to scratch.')

    def _radio_clicked(self, event):
        self._selector.rectprops['facecolor'] = self._colors[event]

    def _check_clicked(self, event):
        print('Event: ', event)
        self.rax.remove()
        self.rax = self.fig.add_axes([0.90, 0.25, 0.09, 0.2], frameon=False)
        self.rax.set_title('Current \ncomponent', size='x-small')
        rlabels = self._peaks[np.where(self.check.get_status())]
        self.radio = RadioButtons(self.rax, rlabels)
        plt.draw()

    def measure_flux(self, xmin, xmax, iters=1):
        idx = np.where((self.wave > xmin) & (self.wave < xmax))
        wav = self.wave[idx]
        vel = wl_to_v(wav, self.refwave)
        dat = self.interp[idx]  # - 1
        fluxes, vmaxs, vmins, fwhms = [], [], [], []
        bhms, rhms, asymmetry = [], [], []
        fmax, fmin = [], []
        if self.errs is None:
            iters = 1
        for i in range(iters):
            perturb = np.array(
                [np.random.normal(scale=e) for e in self.errs[idx]])
            if i > 0:
                pertdata = dat + perturb
            else:
                pertdata = dat
            # fluxes.append(((pertdata - 1) * self._wave_diffs[idx]).sum())
            fluxes.append(((pertdata) * self._wave_diffs[idx]).sum())
            vmaxs.append(vel[dat.argmax()])
            vmins.append(vel[dat.argmin()])
            cumflux = np.cumsum(pertdata - 1)
            # cumflux = np.cumsum(pertdata)
            q05 = vel[np.absolute(cumflux / cumflux.max() - 0.05).argmin()]
            q50 = vel[np.absolute(cumflux / cumflux.max() - 0.50).argmin()]
            q95 = vel[np.absolute(cumflux / cumflux.max() - 0.95).argmin()]
            A = (q95 - q50) / (q50 - q05)
            asymmetry.append(A)
            fwidx = np.where(pertdata - 1 > (pertdata - 1).max()/2)[0]
            bhm = vel[fwidx.min()]
            rhm = vel[fwidx.max()]
            bhms.append(bhm)
            rhms.append(rhm)
            fmax.append((pertdata-1).max())
            fmin.append((pertdata-1).min())
            # fmax.append((pertdata).max())
            # fmin.append((pertdata).min())
            fwhms.append(rhm-bhm)
        return fluxes, vmaxs, vmins, fwhms, bhms, rhms, asymmetry, fmin, fmax

    def absolute_flux(self, xmin=None, xmax=None, iters=1):
        afs = []
        afarray = (self.interp - 1) * self._cont
        aferrar = (self.interr) * self._cont
        if xmin:
            afarray[self.wave < xmin] = 0
        if xmax:
            afarray[self.wave > xmax] = 0
        if self.errs is None:
            iters = 1
        for i in range(iters):
            if i == 0:
                pertdata = afarray
            else:
                perturb = np.array(
                    [np.random.normal(scale=e) for e in np.absolute(aferrar)])
                pertdata = afarray + perturb
            afs.append((afarray * self._wave_diffs).sum())
        self.summary_dict['AbsFlux'] = np.percentile(afs, [2.5, 16, 50, 84, 97.5])
        return self.summary_dict['AbsFlux']  # np.atleast_1d(afs)


    def fit_red(self, xmin, xmax):
        self._selector.rectprops['facecolor'] = 'tab:red'
        self._extra_plots['redspan'] = self.ax.axvspan(
            xmin, xmax, color=self._colors['Red'], alpha=.5, zorder=0)
        flux, vpeak, vmin, fwhm, bhms, rhms, A_qs, fmin, fmax = \
            self.measure_flux(xmin, xmax, iters=1000)
        self.summary_dict['Red'] = {
            'range': (xmin, xmax),
            'flux': np.percentile(flux, [2.5, 16, 50, 84, 97.5]),
            'fmax': np.percentile(fmax, [2.5, 16, 50, 84, 97.5]),
            'vpeak': np.percentile(vpeak, [2.5, 16, 50, 84, 97.5]),
            'fwhm': np.percentile(fwhm, [2.5, 16, 50, 84, 97.5]),
            'blue_at_half_width': np.percentile(bhms, [2.5, 16, 50, 84, 97.5]),
            'red_at_half_width': np.percentile(rhms, [2.5, 16, 50, 84, 97.5]),
            'Asymmetry': np.percentile(A_qs, [2.5, 16, 50, 84, 97.5]),
        }
        print('Fitting red peak')
        print(xmin, xmax)

    def fit_blue(self, xmin, xmax):
        self._extra_plots['bluespan'] = self.ax.axvspan(
            xmin, xmax, color=self._colors['Blue'], alpha=.5, zorder=0)
        flux, vpeak, vmin, fwhm, bhms, rhms, A_qs, fmin, fmax = \
            self.measure_flux(xmin, xmax, iters=1000)
        self.summary_dict['Blue'] = {
            'range': (xmin, xmax),
            'flux': np.percentile(flux, [2.5, 16, 50, 84, 97.5]),
            'fmax': np.percentile(fmax, [2.5, 16, 50, 84, 97.5]),
            'vpeak': np.percentile(vpeak, [2.5, 16, 50, 84, 97.5]),
            'fwhm': np.percentile(fwhm, [2.5, 16, 50, 84, 97.5]),
            'blue_at_half_width': np.percentile(bhms, [2.5, 16, 50, 84, 97.5]),
            'red_at_half_width': np.percentile(rhms, [2.5, 16, 50, 84, 97.5]),}
        print("The following has been added/overwritten"
              + " in the summary_dict['Blue'].")
        return self.summary_dict['Blue']

    def fit_valley(self, xmin, xmax):
        flux, vpeak, vmin, fwhm, bhms, rhms, A_qs, fmin, fmax = \
            self.measure_flux(xmin, xmax, iters=1000)
        self._extra_plots['Valley'] = self.ax.axvspan(
            xmin, xmax, color=self._colors['Valley'], alpha=5.)
        self.summary_dict['Valley'] = {
            'range': (xmin, xmax),
            'minflux': np.percentile(fmin, [2.5, 16, 50, 84, 97.5]),
            'vmin': np.percentile(vmin, [2.5, 16, 50, 84, 97.5])}
        # print(xmin, xmax)
        print('Fitting valley')
        print("The following has been added/overwritten"
              + " in the summary_dict['Valley'].")
        return self.summary_dict['Valley']

    def fit_active(self, xmin, xmax):
        if self.radio.value_selected == 'Blue':
            self.fit_blue(xmin, xmax)
        elif self.radio.value_selected == 'Red':
            self.fit_red(xmin, xmax)
        elif self.radio.value_selected == 'Valley':
            self.fit_valley(xmin, xmax)

    fitfuncs = {'Red': fit_red, 'Blue': fit_blue, 'Valley': fit_valley}

    def save_summary(self):
        # TODO: Implement something.
        pass

    @property
    def _wave_diffs(self):
        diffs = self.wave[1:] - self.wave[:-1]
        diffs = np.append(diffs, diffs[-1])
        return diffs

    @property
    def refwave(self):
        return 1215.67 * (1 + self.z)

    @property
    def ref_wl(self):
        return 1215.67 * (1 + self.z)

    def __call__(self):
        self.fig.show()


"""
TODO: The things Max wished for
FWHM_red DONE
FWHM_blue DONE
F_valley / F_peak DONE
L_red / L_blue WONTDO
L_red(f > 1) DONE
F_blue / F_red DONE
A_red (e.g., (q_95 - q_50) / (q_50 - q_5)  DONE
v_red DONE
v_blue DONE
TODO: EW Full line


with
FWHM – FWHM of peak
F – flux of peak
L – integrated flux of peak
q – percentile
v – peak position
"""

