#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import json
import textwrap as tw
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
from astropy.table import Table, Column
from astropy.io import ascii
from astropy.modeling import fitting, models
import astropy.constants as c
import astropy.units as u
import lmfit as lm
from .helper_functions import wl_to_v, v_to_wl, v_to_deltawl, air_to_vacuum
from .helper_functions import vacuum_to_air, equivalent_width
from .linelists import lislines, wlsdict, MWlines
import skylines.uves_sky_atlas as usa

class BaseGUI(object):
    def __init__(self):
        raise NotImplementedError("This class is still a work in progress")


class GalaxySpectrum(object):
    """ Insert docstring here.
    """
    objname = None
    datatable = Table()
    z = 0.0
    frequnit = u.GHz
    waveunit = u.Angstrom
    velunit = u.km/u.second
    dataunit = None
    transitions = None
    preferred_flux = 'flam'

    def read_data(self, datafile):
        """ So far only calls ascii.read() with no fancy settings.
        """
        datatable = ascii.read(datafile)
        # Make sure all columns have correct datatype
        for cn in datatable.columns:
            c = datatable.columns[cn]
            try:
                datatable[cn] = np.float64(c)
            except:
                raise TypeError(
                    "Column {} could not be converted to floats.".format(cn))
        self.datatable = datatable
        if not 'flux' in self.datatable.colnames:
            if self.preferred_flux in self.datatable.colnames:
                self.datatable[self.preferred_flux].name = 'flux'
            elif 'flam' in self.datatable.colnames:
                self.datatable['flam'].name = 'flux'
            elif 'fnu' in self.datatable.colnames:
                self.datatable['fnu'].name = 'flux'
            else:
                raise KeyError("No flux column recognized from name")

    def set_units(self, units=None):
        set_units(self.datatable, units=units)

    def add_transition(self, transname, ref_transition=None):
        t = add_transition(self, transname, ref_transition)
        return t

    def rebin(self, factor):
        """ CAUTION - changes the data table of this object."""
        t = rebin(self.datatable, factor)
        self.datatable = t
        print("Data have been rebinned by a factor of {}".format(factor))

    def smooth_data(self, width=1):
        return np.convolve(self.data, np.ones(width) / width, mode='same')

    def smooth_errs(self, width=1):
        return np.convolve(self.errs, np.ones(width) / width, mode='same')

    def velocity(self, cenwave):
        v = wl_to_v(self.wave.to(u.Angstrom).value, cenwave) * u.km / u.second
        return v

    @property
    def wave(self):
        """Docstring goes here"""
        return self.datatable['wave'].to(self.waveunit)

    @property
    def obswave(self):
        """Docstring goes here"""
        try:
            return self.datatable['obswave'].to(self.waveunit)
        except KeyError:
            print("Dataset does not distinguish true and obs wavelength")
            return self.data

    @property
    def data(self):  # TODO Maybe make this selectable instead of hardcoded?
        try:
            return self.datatable['flux'].quantity
        except KeyError:
            pass
        try:
            return self.datatable['fnu'].quantity
        except KeyError:
            raise KeyError('Flux must be either "flux", "flam" or "fnu".')

    @property
    def errs(self):
        return self.datatable['noise'].quantity

    @property
    def frequency(self):
        return (1 / self.wave.to(u.m) * c.c).to(self.frequnit)
        #return freq

    @property
    def restwave(self):
        return (self.wave / (1 + self.z)).to(self.waveunit)

    @property
    def _base_lines(self):
        if self.transitions is None:
            return None
        out = [t.name for t in self.transitions.values() if t.reference_transition is None]
        return out

    @property
    def line_sets(self):
        lsets = {}
        for l in self._base_lines:
            tlines = []
            for t in self.transitions.values():
                if t.reference_transition is None:
                    continue
                if t.reference_transition.name == l:
                    tlines.append(t.name)
            lsets[l] = tlines
        return lsets

    @property
    def summary_dict(self):
        sd = {}
        # sd['datatable'] = self.datatable
        sd['waveunit'] = str(self.waveunit) if self.waveunit else ""
        sd['velunit'] = str(self.velunit) if self.velunit else ""
        sd['frequnit'] = str(self.frequnit) if self.frequnit else ""
        sd['dataunit'] = str(self.dataunit) if self.dataunit else ""
        sd['transitions'] = {t: self.transitions[t].summary_dict for t in self.transitions}
        sd['galaxyname'] = self.objname
        sd['redshift'] = float(self.z)
        return sd

    def velocity_table(self, theset):
        """ `theset` must e a string naming the base transitions of the desired
        set.
        """
        transitions = self.transitions
        if not theset in self._base_lines:
            raise KeyError("This transition is not the base of any line set")
        T = Table(self.transitions[theset].velocity.reshape(-1, 1))
        T.columns[0].name = 'Velocity'
        alllines = [theset] + self.line_sets[theset]
        for line in alllines:
            colnames = ["{} {}".format(line, i) for i in ['flux', 'errs', 'mask']]
            fcol = Column(transitions[line].data_resampled, name=colnames[0])
            ecol = Column(transitions[line].errs_resampled, name=colnames[1])
            mcol = Column(transitions[line].mask_resampled, name=colnames[2])
            T.add_columns([fcol, ecol, mcol])
        return T

    def save_summary(self, path, file_format='yml'):
        linelist = [t.name for t in self.transitions.values()]
        preamble = """# This is the summary for galaxy {}, containing the lines
        """.format(self.objname) + str(linelist)
        if file_format.lower() in ['y', 'yml', 'yaml']:
            preamble = tw.fill(preamble, width=78, subsequent_indent="# ")
            preamble += '\n'
            with open(path, 'w') as f:
                f.write(preamble+"\n\n")
                yaml.dump(self.summary_dict, f, indent=2)
        elif file_format.lower() in ['j', 'jsn', 'json']:
            with open(path, 'w') as f:
                json.dump(self.summary_dict, f, indent=2)
        else:
            raise KeyError('File format must be yaml or json')

    def load_summary(self, summary):
        self.objname = summary['galaxyname']
        self.z = summary['redshift']
        for t in summary['transitions']:
            T = Transition()


class Transition(object):
    """ This object contains two kinds of information: Intrinsic information
    about the transition itself, and information about the observed propertiues
    of said transition in a given spectrum.

    Intrinsic properties are always given in rest-frame/laboratory wavelengths,
    while observed properties are given in observed wavelengths. This may be a
    little confusing, but it should have some logic to it, and it works.
    """
    # Metadata
    def __init__(self):
        self.name = None
        self.dataunit = u.Unit('')
        self.waveunit = u.Unit('')
        self.galaxyname = None
        self.cont_fit_params = None
        self.cont_fit_include = None
        self.reference_transition = None  # Should be string
        self.masked = False
        self._mask = None
        self.fitted = False
        self.fitter = None  # Placeholder for fitting result
        self.z = 0
        # The following are Base quantities, assumed type is astropy.Quantity
        # Other quantities are numbers, assuming the relevant units are inherited.
        self.vac_wl = None  # np.nan * u.Unit('')  # *Restframe* wavelength in vacuum
        self.wave = None  # np.array([]) * u.Unit('')   # Should later become array of *observed* wavelengths.
        self.data = None  # np.array([]) * u.Unit('')    #  -"- data
        self.errs = None  # np.array([]) * u.Unit('')    #  -"- errors TODO Should this be a Quantity or array?

    @property
    def centroid(self):
        return self.vac_wl.value  # Make more smart one day

    @property
    def obs_centroid(self):
        return self.centroid * (1+self.z)

    @property
    def rest_wave(self):
        return self.wave.value / (1+self.z)

    @property
    def velocity(self):
        return wl_to_v(self.wave.value, self.centroid * (1. + self.z))

    @property
    def thistype(self):
        return(type(self))

    @property
    def wave_resampled(self):
        if self.reference_transition is None:
            return self.wave.value
        else:
            wave = v_to_wl(
                self.reference_transition.velocity,#.value,
                self.centroid
            )
            return wave

    @property
    def velocity_resampled(self):
        if self.reference_transition is None:
            return self.velocity
        else:
            return self.reference_transition.velocity

    @property
    def data_resampled(self):
        if not self.reference_transition:
            dr = self.data.value
        elif self.data is None:
            dr = None
        else:
            dr = np.interp(self.velocity_resampled, self.velocity, self.data)
            dr = dr.value
        return dr

    @property
    def errs_resampled(self):
        # NB! This does not strictly treat errors correctly, but it is good
        # enough for our purposes.
        if self.reference_transition is None:
            return self.errs.value
        else:
            newerrs = np.interp(
                self.velocity_resampled, self.velocity, self.errs.value
            )
            return newerrs

    @property
    def mask(self):
        if self.data is None:
            return None
        if not self.masked:
            return np.zeros_like(self.data.value, dtype='bool')
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask
        self.masked = True
    @property
    def mask_resampled(self):
        if self.reference_transition is None:
            return self.mask
        else:  # if isinstance(self.reference_transition, Transition):
            floatmask = self.mask.astype(float)
            newfloatmask = np.around(np.interp(
                self.velocity_resampled, self.velocity, floatmask
            )).astype(bool)
            return newfloatmask

    def plot(self, ax=None, smooth=1, maskalpha=1, showmasked=True, **kwargs):
        """ Insert docstring
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        if self.mask is None:
            mask = np.zeros_like(self.data).astype(bool)#.value
        else:
            mask = self.mask
        invmask = np.invert(mask)
        data = np.convolve(self.data.value, np.ones(smooth)/smooth, mode='same')
        errs = np.convolve(self.errs.value, np.ones(smooth)/smooth, mode='same')
        plotdata = np.ma.masked_array(data, mask)
        invdata = np.ma.masked_array(data, invmask)
        ploterrs = np.ma.masked_array(errs, mask)
        inverrs = np.ma.masked_array(errs, invmask)
        p = ax.plot(
            self.velocity, plotdata, label=self.name,
            drawstyle='steps-mid', linestyle='-', **kwargs,
        )[0]
        if showmasked:
            ax.plot(
                self.velocity, invdata, label='_nolabel',
                drawstyle='steps-mid', linestyle='--',
                color=p.get_color(), alpha=maskalpha
            )
        return p

    def load_summary(self, path):
        """ Alias for load_line_summary()"""
        self.load_line_summary(path)

    @property
    def summary_dict(self):
        # print(self.reference_transition)
        if self.reference_transition:
            refname = self.reference_transition.name
        else:
            refname = None

        if self.fitted:
            fitpars = {
                'slope': float(
                    self.cont_fit_params['slope'].value),
                'intercept': float(
                    self.cont_fit_params['intercept'].value)}
        else:
            fitpars = None

        D = {
            'name': self.name,
            'galaxyname': self.galaxyname,
            'reference_transition_name': refname,
            'vac_wl': {
                'value': self.vac_wl.value.tolist(),
                'unit': self.vac_wl.unit.to_string()
            },
            # Exclude resampled versions? Redundant but useful.
            'wave': None if self.data is None else self.wave.value.tolist(),
            'wave_resampled':
                None if self.data is None else self.wave_resampled.tolist(),
            'velocity': None if self.data is None else self.velocity.tolist(),
            'velocity_resampled':
                None if self.data is None else self.velocity_resampled.tolist(),
            'data': None if self.data is None else self.data.value.tolist(),
            'data_resampled':
                None if self.data is None else self.data_resampled.tolist(),
            'errs': None if self.errs is None else self.errs.value.tolist(),
            'errs_resampled':
                None if self.errs_resampled is None else self.errs_resampled.tolist(),
            'mask': None if self.mask is None else self.mask.tolist(),
            'mask_resampled':
                None if self.mask is None else self.mask_resampled.tolist(),
            'cont_fit_include': np.array(self.cont_fit_include).tolist(),
            # None if not self.fitted else self.cont_fit_include.tolist(),
            'cont_fit_params': fitpars,  # self.cont_fit_params,  # fitpars,
            'z': float(self.z),
        }
        return D

    def save_summary(self, path, file_format='yml'):
        if file_format in ['y', 'yml', 'yaml']:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(self.summary_dict, f, indent=2)
        if file_format in ['j', 'jsn', 'json']:
            import json
            with open(path, 'w') as f:
                json.dump(self.summary_dict, f, indent=2)

    def load_line_summary(self, summary):
        """ Summary datatype is a dict, loaded from yaml file."""
        self.name = summary['name']
        params = [
            "name", "vac_wl", "z",
            "wave", "data", "errs", "mask",
            "reference_transition_name",
            "cont_fit_include", "cont_fit_params",
            "continuum_fit_include", "continuum_fit_params",  # For backward compatibility
            "waveunit", "dataunit",
        ]

        for par in params:
            # print(par, par in summary.keys())
            if par not in summary.keys():
                continue
            if par == "mask":
                self._mask = np.array(summary["mask"])
                self.masked = True
            elif par == "vac_wl":
                self.vac_wl = \
                    summary['vac_wl']['value'] * u.Unit(summary['vac_wl']['unit'])
            elif par == "dataunit":
                self.dataunit = u.Unit(summary[par])
                self.errsunit = u.Unit(summary[par])
            elif par == "waveunit":
                self.wave *= u.Unit(summary[par])
            elif "continuum" in par:
                oldpar = par
                par = par.replace("inuum", "")
                self.__dict__.update({par: summary[oldpar]})  # [par] == summary[par]
            else:
                self.__dict__.update({par: summary[par]})  # [par] == summary[par]

        self.data = np.array(self.data) * self.dataunit
        self.errs = np.array(self.errs) * self.dataunit
        self.wave = np.array(self.wave) * self.waveunit





class SpecView(object):
    """ Docstring goes here.
    """

    alt_xaxis = None
    altaxis_type = None
    skyatlas = None
    skylines_visible = False
    _metal_absorption = None
    _absorption_visible = False
    _sky_lines = None
    _sky_flux_limit = 1  # Sane default

    def __init__(self, galaxy, ax=None, data=None, label='Data'):
        self.data = galaxy.datatable
        self.galaxy = galaxy
        self.ref_wl = None
        if ax is None:
            fig, ax = plt.subplots(1)
        self.dataplot = ax.plot(
            self.data['wave'], self.data['flux'],
            lw=1.5,
            drawstyle='steps-mid',
            label=label,
            color='black',
        )[0]
        self.errplot = None
        ax.axhline(0, ls='--', color='black')
        ax.set_ylabel("Flux [{}]".format(self.data['flux'].unit))
        ax.set_xlabel("Observed $\lambda$ [{}]".format(self.data['wave'].unit))
        self._smooth_width = 1  # No smoothing by default
        self.ax = ax

    def toggle_sky_lines(self, min_flux=1):
        if min_flux != self._sky_flux_limit:  # Rebuild if new value passed
            self._sky_flux_limit = min_flux
            self.skyatlas = None
            self.skylines_visible = False
            for l in self._sky_lines:
                l.remove()
                del l
            self._sky_lines = None

        if self.skyatlas is None:
            print("Loading UVES sky line atlas...")
            self.skyatlas = usa.UvesSkyAtlas()
            self.skyatlas.load_lines()

        if self.skylines_visible:
            for l in self._sky_lines:
                l.set_visible(False)
            self.skylines_visible = False
        elif self._sky_lines is not None:
            for l in self._sky_lines:
                l.set_visible(True)
            self.skylines_visible = True
        else:
            print("Building sky line plot...")
            self._sky_lines = []
            for i, t in enumerate(self.skyatlas.line_collections.keys()):
                color = 'C{}'.format(i)
                for l in self.skyatlas.line_collections[t]:
                    if t == '800U':  # Hardcoded ugly hack bcs Srsly WTF ESO/UVES?!
                        self.skyatlas.line_collections[t]['CENTER'] = \
                            self.skyatlas.line_collections[t]['LAMBDA_AIR']
                    if l['FLUX'] > min_flux:
                        ll = self.ax.axvline(l['CENTER'], color=color)
                        self._sky_lines.append(ll)
            self.skylines_visible = True

    def toggle_errors(self):
        if self.errplot is None:
            self.errplot = self.ax.plot(
                self.data['wave'], self.data['noise'],
                lw=1.5,
                drawstyle='steps-mid',
                label='noise',
                color='0.8',
                zorder=0.
            )[0]
            self.smooth_width(self._smooth_width)
        else:
            self.errplot.remove()
            self.errplot = None
            #plt.draw()

    def toggle_restframe_xaxis(self):
        if self.altaxis_type == 'restframe':
            self.altx.remove()
            self.altaxis_type = None
        else:
            try:
                self.altx.remove()
            except:
                pass
            ax = self.ax
            altx = ax.twiny()
            self.altx = altx
            altx.set_xlim(self.restwave_lims.value)
            altx.set_xlabel(
                '{} Restframe $\lambda$ [{}]'.format(
                    self.galaxy.objname, self.galaxy.waveunit))
            self.altaxis_type = 'restframe'
            self._fix_zorder()

    def toggle_frequency_xaxis(self):
        if self.altaxis_type == 'freq':
            self.altx.remove()
            self.altaxis_type = None
        else:
            try:
                self.altx.remove()
            except:
                pass
            ax = self.ax
            altx = ax.twiny()
            self.altx = altx
            altx.set_xlim(self.freq_lims.value)
            altx.set_xlabel(
                '{} frequency [{}]'.format(
                    self.galaxy.objname, self.galaxy.frequnit))
            self.altaxis_type = 'freq'
            self._fix_zorder()

    def toggle_velocity_xaxis(self):
        """Toggle upper velocity axis.

        Prompts for a reference wavelength if the ref_wl attribute of the
        `SpecView` instance is not set.
        """
        if self.ref_wl is None:
            self.ref_wl = float(input('Please enter reference wavelength > '))

        if self.altaxis_type == 'vel':
            self.altx.remove()
            self.altaxis_type = None
        else:
            try:
                self.altx.remove()
            except:
                pass
            ax = self.ax
            altx = ax.twiny()
            self.altx = altx
            altx.set_xlim(self.vel_lims.value)
            altx.set_xlabel(
                '{} Velocity [{}]'.format(
                    self.galaxy.objname, self.galaxy.velunit))
            self.altaxis_type = 'vel'
            self._fix_zorder()

    def toggle_metal_absorption(self, col1='C0', col2='C2'):
        # TODO Implement a "wave type" keyword to allow for velocity, frequency
        # etc. to occur on primary x-axis(for subclassing purposes)
        # If not drawn (and consequentially not visible), draw and set visible:
        if self._metal_absorption is None:
            print("Populating absorption line list...")
            # from linelists import MWlines
            self._metal_absorption = {}
            self._metal_annotations = {}
            for absln in MWlines.keys():
                control_waverange = \
                    ((MWlines[absln] * (1.+self.galaxy.z)
                      > self.galaxy.wave.value.min()) &
                     (MWlines[absln] * (1.+self.galaxy.z)
                      < self.galaxy.wave.value.max()))
                if control_waverange:
                    self._metal_absorption[absln] = self.ax.axvline(
                        MWlines[absln] * (1. + self.galaxy.z),
                        #color='lightgray',
                        color=col1,
                        linestyle=':'
                    )
                    self._metal_annotations[absln] = self.ax.annotate(
                        absln,
                        xy=(MWlines[absln] * (1 + self.galaxy.z), 0.85),
                        xycoords=('data', 'axes fraction'),
                        color=col1,
                        backgroundcolor='w',
                        rotation=90,
                        annotation_clip=True,
                    )

                control_waverange_MW = (
                    (MWlines[absln] > self.galaxy.wave.value.min()) &
                    (MWlines[absln] < self.galaxy.wave.value.max()))
                if control_waverange_MW:
                    self._metal_absorption[absln+'_MW'] = self.ax.axvline(
                        MWlines[absln],
                        color=col2,
                        linestyle=':',
                    )
                    self._metal_annotations[absln+'_MW'] = self.ax.annotate(
                        absln+' MW',
                        xy=(MWlines[absln], 0.85),
                        xycoords=('data', 'axes fraction'),
                        color=col2,
                        backgroundcolor='w',
                        rotation=90,
                        annotation_clip=True,
                    )
            self._absorption_visible = True
        # If already drawn and visible, hide.
        elif self._absorption_visible:
            for absln in self._metal_absorption.keys():
                self._metal_absorption[absln].set_visible(False)
                self._metal_annotations[absln].set_visible(False)
            self._absorption_visible = False
        # If already drawn, but hidden, set to visible:
        else:
            for absln in self._metal_absorption:
                self._metal_absorption[absln].set_visible(True)
                self._metal_annotations[absln].set_visible(True)
                # self._metal_absorption[absln+'_MW'].set_visible(True)
                # self._metal_annotations[absln+'_MW'].set_visible(True)
                self._absorption_visible = True
        return

    def smooth_width(self, width=1):
        """ Smooths the plotted data by the width set; default is 1.
        Width is measured in data bins.
        """
        self._smooth_width = width
        if width < 1:
            warnings.warn(
                'Kernel width cannot be less than 1 \nIgnoring.',
                RuntimeWarning

            )
            return
        width = max(round(width), 1)
        self.dataplot.set_data(
            self.dataplot.get_data()[0],
            self.galaxy.smooth_data(width)
        )
        if self.errplot is not None:
            self.errplot.set_data(
                self.errplot.get_data()[0],
                self.galaxy.smooth_errs(width)
            )

    @property
    def freq_lims(self):
        out = (
            c.c / (self.ax.get_xbound() * self.data['wave'].unit)
        ).to(self.galaxy.frequnit)
        #print(out)
        self.altx.set_xlim(out.value)#[0], out[1])
        return out

    @property
    def restwave_lims(self):
        wave = (
            np.array(self.ax.get_xbound()) * self.galaxy.waveunit
        ).to(u.Angstrom)
        restwave = wave / (1 + self.galaxy.z)
        self.altx.set_xlim(restwave.value)
        return restwave

    @property
    def vel_lims(self):
        ref_wl = self.ref_wl
        wave = (
            np.array(self.ax.get_xbound()) * self.galaxy.waveunit
        ).to(u.Angstrom)
        ref_wl = (ref_wl * self.galaxy.waveunit).to(u.Angstrom)
        out = (
            wl_to_v(wave, ref_wl) * (u.km / u.second)
        ).to(self.galaxy.velunit)
        self.altx.set_xlim(out.value)
        return out

    def _fix_zorder(self):
        """ Addresses an annoying behavior introduced in Matplotlib 2.1, where
        the latest added axes steals events. So manually, we need to fix the
        zorder such that the 'mother' axes is on top.
        Remove this if upstream matplotlib comes up with a fix. Very irritating
        regression.
        """
        self.ax.set_zorder(self.altx.get_zorder()+1)


class SimpleFitGUI(SpecView):
    """ Simple interactive GUI for fitting a model to a 1D spectrum.
    By default, the model is a linear one, but any lmfit.Model object can be
    passed (still needs testing though).

    Usage:

    - Initialize the gui with data, model etc.: `fitter = SimpleFitGUI(...)`
    - Call it to open the GUI: `fitter()`
    - Click and drag to mark regions, and the fit will be updated in real time.
    - At any time, use `fitter.report()` to get a summary of the current best
      fit.
    - Press the OK button to have the data normalized by the current best-fit
      model saved to the `Transition` object passed to the fitter.
    - Press the Clear button to reset interface (TODO also reset parameters?)
    """

    # Class attributes:
    model = None  # For simple subclassing to specialized cases -> save effort

    # Now, init and all the instance vars
    def __init__(self, galaxy, transition, model=None, smooth=6, show_lines=True):
        """
        Parameters
        ----------
        galaxy : GalaxySpectrum
            The galaxy spectrum to fit.
        transition : Transition
            Transition to work on. This must (?) be in the `transitions` dict of
            the GalaxySpectrum passed to the GUI.
        model : lmfit.models.Model
            The model to fit to the data. By default a linear function.
        """
        self.galaxy = galaxy
        self.z = galaxy.z
        if smooth is None:
            kernelwidth = 1
        else:
            kernelwidth = max(1, smooth)  # Kernel cannot be less than 1.
        self.plotdata = np.convolve(
            self.galaxy.data, np.ones(kernelwidth)/kernelwidth, mode='same')
        self.transition = transition
        self.centroid = transition.centroid * (1 + galaxy.z)
        self.idx = np.zeros_like(self.galaxy.data.value).astype(bool)
        if model is None:
            if self.model is None:
                self.model = lm.models.LinearModel(name='cont')
        else:
            self.model = model
        self.params = self.model.make_params(slope=0.01, intercept=0)
        self.fitresults = {a:np.nan for a in self.model.param_names}  # Using this?
        self.modelplot = None
        self.model_fitted = None
        self.fitted = False
        self.fig, self.ax = plt.subplots(1)
        self._build_plot()
        if show_lines:
            self.toggle_metal_absorption()
            #add_line_markers(self, ls='--')

    def _build_plot(self):
        """ Name should be self explaining. Separated from `__init__` for
        simplicity, and because I suspected this function could come in handy
        for debugging and other tinkering purposes later.
        """
        self.ax.plot(
            self.galaxy.wave, self.plotdata, 'k',
            drawstyle='steps-mid',
            label=self.transition.name
        )
        xlow, xup = self.centroid - 10 * (1+self.z), \
            self.centroid + 10 * (1+self.z)
        self.ax.set_xlim(xlow, xup)
        self.ax.fill_between(
            self.galaxy.wave.value,
            self.galaxy.errs.data,
            color='gray', alpha=.5
        )
        # print(np.median(self.data))
        self.ax.axhline(1, color='k', ls='--')
        self.ax.set_ylim(bottom=0., top=np.nanmedian(self.galaxy.data.value) * 2.)
        axshw = self.fig.add_axes([0.91, 0.82, 0.08, 0.06])
        axclr = self.fig.add_axes([0.91, 0.75, 0.08, 0.06])
        ax_ok = self.fig.add_axes([0.91, 0.68, 0.08, 0.06])
        self.shwbu = Button(axshw, 'Print')
        self.clrbu = Button(axclr, 'Clear')
        self.ok_bu = Button(ax_ok, 'OK')
        self.shwbu.on_clicked(self._on_show_button)
        self.clrbu.on_clicked(self._on_clr_button)
        self.ok_bu.on_clicked(self._on_ok_button)
        if self.centroid is not None:
            self.ax.axvline(
                x=self.centroid, linestyle='--', color='gray', alpha=.6, lw=1.5)
        # if self.transition is not None:
        #     self.ax.legend(loc='lower left')
        self.ax.annotate(
            "{} {}".format(self.galaxy.objname, self.transition.name),
            (0.1, 0.1), xycoords='axes fraction',
        )
        s = 'Click and drag to mark ranges to include in fit.'# \n \
        helptext = self.ax.text(
            .5, .95, s,
            bbox=dict(
                fc='0.9', ec='0.8', alpha=.9, boxstyle='round'
            ),
            horizontalalignment='center',
            verticalalignment='top',
            transform=self.ax.transAxes
        )
        span = np.nanmean(self.galaxy.wave.diff().value) * 5
        self.span = SpanSelector(
            self.ax, self._onselect, 'horizontal', minspan=span,
        )

    def __call__(self):
        self.fig.show()

    def _onselect(self, vmin, vmax):
        ###============================================
            # Set everything between vmin and vmax to True.
        ###============================================
        mask = np.where((self.galaxy.wave.value > vmin) &
                        (self.galaxy.wave.value < vmax))
        self.idx[mask] = True
        self.spans = self.ax.axvspan(
            vmin, vmax, color='C1', alpha=1.0, zorder=0, picker=True)
        self._fit_and_update()
        self.fitted = True
        self.fig.canvas.draw()

    def _fit_and_update(self):
        # If already fitted, just update existing ModelResult
        weights = self.galaxy.errs[self.idx].value ** -1
        #weights /= weights.sum()
        self.weights = weights
        if not self.fitted:
            self.model_fitted = self.model.fit(
                self.galaxy.data[self.idx].value,
                self.params,
                x=self.galaxy.wave[self.idx].value,
                weights=weights,  #self.galaxy.errs[self.idx].value**2,
            )
        else:
            self.model_fitted.fit(
                self.galaxy.data[self.idx].value,
                self.params,
                x=self.galaxy.wave[self.idx].value,
                weights=weights,  # self.galaxy.errs[self.idx].value**2,
            )
        # If best-fit already plotted, just update that one.
        if self.modelplot is None:
            self.modelplot = self.ax.plot(
                self.galaxy.wave.value,
                self.galaxy.wave.value * self.model_fitted.best_values['slope'] \
                + self.model_fitted.best_values['intercept'],
                color='C0',
            )[0]
        else:
            self.modelplot.set_data(
                self.galaxy.wave.value,
                self.galaxy.wave.value * self.model_fitted.best_values['slope'] \
                    + self.model_fitted.best_values['intercept'],
            )
        plt.draw()
        return  # self.model_fitted

    def _on_ok_button(self, event):
        # Only interested in immediate surroundings
        mask_bool = (  # Use np.where or not?
            (self.galaxy.restwave.value > self.transition.centroid - 30)
            & (self.galaxy.restwave.value < self.transition.centroid + 30))
        mask = np.where(mask_bool)
        # Cut out wavelength range
        self.transition.wave = self.galaxy.wave[mask]
        # Divide by the best-fit model in a way independent of which model we
        # are using, for better subclassing later.
        best_fit_eval = self.model_fitted.eval(
            self.model_fitted.params,
            x=self.transition.wave.value) * self.galaxy.datatable['flux'].unit
        self.transition.data = self.galaxy.data[mask] / best_fit_eval
        self.transition.errs = self.galaxy.errs[mask] / best_fit_eval
        self.transition.z = self.galaxy.z
        self.transition.fitted = True
        self.transition.fitter = self
        self.transition.cont_fit_params = self.model_fitted.params
        self.transition.cont_fit_include = np.where(self.idx & mask_bool)[0]
        for spans in self.ax.patches:
            spans.set_color('#FBB117')
        self.galaxy.transitions[self.transition.name] = self.transition
        print(
            "Normalized data saved to Transition {} in spectrum {}".format(
                self.transition.name, self.galaxy.objname
            )
        )

    def _on_show_button(self, event):
        self.report()

    def _on_clr_button(self, event):
        # print(self.fitted)
        if self.fitted:
            self.ax.patches = []
            self.ax.lines.pop()
            self.idx[:] = False
            self.fig.canvas.draw()
            self.fitted = False
            self.modelplot = None
        else:
            print("Nothing to clear up")

    def report(self):
        """ Prints the fitting results, errors and statistics to stdout."""
        if self.model_fitted is not None:
            print(self.model_fitted.fit_report())
        else:
            print("No fits performed so far")


class SimpleMaskGUI(SimpleFitGUI):
    """ Interactively set map on Transition data.
    masks being averaged like the rest at resampling.

    (Possibly more flexibility to be added later)
    """
    def __init__(self, transition, smooth=6, showlines=True):
        """
        Parameters
        ----------
        transition : Transition
            Transition to work on. This must (?) be in the `transitions` dict of
            the GalaxySpectrum passed to the GUI.
        smooth : int
            number of pixels by with to smooth the data for presentation.
        """
        self.transition = transition
        self.z = transition.z
        if smooth is None:
            kernelwidth = 1
        else:
            kernelwidth = max(1, smooth)  # Kernel cannot be less than 1.
        self.kernelwidth = kernelwidth
        self.data = transition.data#.value
        self.wave = transition.velocity#.value
        self.mask = np.zeros_like(
            self.transition.velocity).astype(bool)
        self.fig, self.ax = plt.subplots(1, figsize=(8, 5))
        self._build_plot()
        if showlines:
            add_line_markers(self, ls='--', wave='vel')

    def _onselect(self, vmin, vmax):
        idx = np.where((self.transition.velocity > vmin) &
                       (self.transition.velocity < vmax))
        self.mask[idx] = True
        self.ax.axvspan(vmin, vmax, color='C2', alpha=0.5, zorder=0)
        self.fig.canvas.draw()

    def _on_ok_button(self, event):
        self.transition.mask = self.mask
        self.ax.patches = []
        self.ax.lines = []  #.pop()
        self.transition.plot(self.ax, smooth=self.kernelwidth, color='k')
        self.ax.axhline(1, ls='--', color='k', lw=.8)

    def _build_plot(self):
        """ Name should be self explaining. Separated from `__init__` for
        simplicity, and because I suspected this function could come in handy
        for debugging and other tinkering purposes later.
        """
        self._dataplot = self.ax.plot(
            self.transition.velocity,
            self.plotdata, 'k',
            drawstyle='steps-mid',
            label=self.transition.name
        )[0]
        self.ax.set_xlim(-3000, 3000)
        self.ax.fill_between(
            self.transition.velocity,
            self.transition.errs,#.value,
            color='gray', alpha=.5
        )
        # print(np.median(self.data))
        self.ax.axhline(1, color='k', ls='--', lw=.8)
        self.ax.set_ylim(bottom=0., top=np.nanmedian(self.data.value) * 3.)
        axshw = self.fig.add_axes([0.91, 0.82, 0.08, 0.06])
        axclr = self.fig.add_axes([0.91, 0.75, 0.08, 0.06])
        ax_ok = self.fig.add_axes([0.91, 0.68, 0.08, 0.06])
        self.shwbu = Button(axshw, 'Print')
        self.clrbu = Button(axclr, 'Clear')
        self.ok_bu = Button(ax_ok, 'OK')
        self.shwbu.on_clicked(self._on_show_button)
        self.clrbu.on_clicked(self._on_clr_button)
        self.ok_bu.on_clicked(self._on_ok_button)
        self.ax.set_title("{} {}".format(
            self.transition.galaxyname, self.transition.name))
        s = 'Click and drag to mark ranges to mask out.'# \n \
        helptext = self.ax.text(
            .5, .95, s,
            bbox=dict(
                fc='white', ec='0.8', alpha=.9, boxstyle='round'
            ),
            horizontalalignment='center',
            verticalalignment='top',
            transform=self.ax.transAxes
        )
        span = np.diff(self.transition.velocity).mean() * 5
        self.span = SpanSelector(
            self.ax, self._onselect, 'horizontal', minspan=span,)

    @property
    def plotdata(self):
        pd = np.ma.masked_array(self.data, self.mask)
        if self.kernelwidth > 1:
            pd = np.convolve(
                pd, np.ones(self.kernelwidth)/self.kernelwidth, mode='same')
        return pd


def rebin(table, factor):
    """ Expects column names "wave", "flux", "noise"
    """
    remainder = len(table) % factor
    if remainder > 0:
        table = table[:-remainder]
    num_groups = len(table) / factor
    groups = np.arange(num_groups).repeat(factor)
    wave_bin = table['wave'].group_by(groups).groups.aggregate(np.nanmean)
    flux_bin = table['flux'].group_by(groups).groups.aggregate(np.nanmean)
    errs_bin = (table['noise']**2).group_by(groups).groups.aggregate(np.sum)
    errs_bin = np.sqrt(errs_bin) / factor
    return Table([wave_bin, flux_bin, errs_bin])


def plot_lines(spectrum, ax=None, smooth=False, binning=False):
    """This is just automated production of a specific plot,
    nothing generalizable.
    """
    hislines = [
        # 'Si IV 1122',
        'Si III 1206',
        'Si IV 1393',
        'Si IV 1402'
    ]
    #fig, axes = plt.subplots(2, 1, sharex=True)
    fig = plt.figure()
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(413, sharex=ax0)
    ax2 = plt.subplot(414, sharex=ax0)

    data = spectrum.datatable['flux']
    wave = spectrum.wave

    if binning:
        # print(len(data))
        remainder = len(data) % binning
        # print(remainder)
        if remainder != 0:
            data = data[:-remainder]
            wave = wave[:-remainder]
        data = data.reshape(-1, binning).nanmean(1)
        wave = wave.reshape(-1, binning).nanmean(1)

    if smooth:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='same')
    refwave = 1215.67 * (1 + spectrum.z)
    vels = wl_to_v(wave.value, refwave)
    idx = np.where((vels > -1500) & (vels < 1500))
    ax0.plot(
        vels[idx], data[idx],
        drawstyle='steps-mid',
        color='darkgray',
        label=r'Ly$\alpha$',
    )

    for line in lislines:
        refwave = wlsdict[line] * (1 + spectrum.z)
        vels = wl_to_v(wave.value, refwave)
        idx = np.where((vels > -1500) & (vels < 1500))
        ax1.plot(
            vels[idx], data[idx],
            drawstyle='steps-mid',
            label=line
        )
    for line in hislines:
        refwave = wlsdict[line] * (1 + spectrum.z)
        vels = wl_to_v(wave.value, refwave)
        idx = np.where((vels > -1500) & (vels < 1500))
        ax2.plot(
            vels[idx], data[idx],
            drawstyle='steps-mid',
            label=line
        )
    for aa in ax0, ax1, ax2:
        aa.axhline(0, color='k', ls='--')
        aa.axvline(0, color='k', ls=':')
    ax1.legend(framealpha=1.)  # .draggable()
    ax2.set_xlabel('Velocity [km s$^{-1}$]')
    ax2.legend(framealpha=1.)  # .draggable()
    ax2.set_xlim(-1500, 1500)
    return


def set_units(intable, units=None, inplace=True):
    """ Sets the proper units for a GalaxySpectrum datatable
    which in turn is an astropy.table).

    Mainly written with the MagE spectra in mind.

    Parameters
    ----------
    intable : astropy.table.Table
        The table to set the units for
    units : list of astropy.units  (optional)
        Units to give the columns. If shorter than number of columns,
        the `len(units)` first columns will be given units.
    """
    if units is None:
        units = np.array([
            u.Angstrom,
            u.erg / (u.cm ** 2 * u.Hz * u.second),
            u.erg / (u.cm ** 2 * u.Hz * u.second),
            u.Angstrom,
            u.erg / (u.cm ** 2 * u.Hz * u.second),
        ])
    # Return modified copy or operate on original input?
    if not inplace:
        table = intable.copy()
    else:
        table = intable

    for i, unit in enumerate(units):
        table.columns[i].unit = unit
    if not inplace:
        return table
    return


def add_transition(galaxy_spectrum, transname, ref_transition=None):
    """ NB! THIS FUNCTION ALTERS THE INPUT OBJECT.

    Convenience function to add transitions from my lists to a
    GalaxySpectrum instance. Of course, one can construct the transitions
    manually and name them anything one wants.

    Parameters
    ----------
    galaxy_spectrum : GalaxySpectrum
        Instance of the GalaxySpectrum class, to which the transition should be
        added.
    transname : str
        Must be in linelists.MWlines
    ref_transition : str
        Transition onto whose velocity scale to resample the other transitions
        AOD and similar per-bin, multi/transition analyzes.
        Must be in galaxy_spectrum.transitions. This function does not check for
        this, but later computations will mess up if the requirement is not met.
    """
    t = Transition()
    t.name = transname
    t.galaxyname = galaxy_spectrum.objname
    t.z = galaxy_spectrum.z if galaxy_spectrum else 0
    # t.vac_wl = MWlines[transname] * galaxy_spectrum.datatable['wave'].unit
    t.dataunit = u.Unit('') if galaxy_spectrum.dataunit is None else galaxy_spectrum.dataunit
    # print(t.dataunit)
    # t.dataunit = dataunit
    t.waveunit = galaxy_spectrum.waveunit if galaxy_spectrum else u.Unit('')
    # print(type(t.waveunit), t.waveunit)
    # t.waveunit = waveunit
    t.vac_wl = (MWlines[transname] * u.Angstrom).to(galaxy_spectrum.waveunit)
    if galaxy_spectrum.transitions is None:
        galaxy_spectrum.transitions = {}
    if ref_transition is not None:
        t.reference_transition = ref_transition
    galaxy_spectrum.transitions[transname] = t
    print("Successfully added the transition {}.\n".format(transname))
    return t


def add_line_markers(view, color1='C0', color2='C2', wave='wave', **kwargs):
    """Parameter `wave` is assumed to be velocity if not specified as 'wave'"""
    for i in MWlines:
        if wave == 'wave':
            gcentroid = MWlines[i] * (1 + view.z)
            mcentroid = MWlines[i]
            halfrange = 50 # * u.Angstrom TODO implement quantity
        else:
            gcentroid = wl_to_v(
                MWlines[i]* (1 + view.z), view.transition.obs_centroid
            )
            mcentroid = wl_to_v(MWlines[i], view.transition.obs_centroid)
            halfrange = 20000
        # print(gcentroid, mcentroid)
        if ((mcentroid > view.transition.obs_centroid - halfrange)
                & (mcentroid < view.transition.obs_centroid + halfrange)):
            view.ax.axvline(mcentroid, color=color1, **kwargs)
            view.ax.annotate(
                i+"_MW", (mcentroid, 0.85), xycoords=('data', 'axes fraction'),
                color=color1, rotation=270, size='x-small',
                annotation_clip=True)
        if (gcentroid > view.transition.obs_centroid - halfrange) \
                & (gcentroid < view.transition.obs_centroid + halfrange):
            view.ax.axvline(gcentroid, color=color2, **kwargs)
            view.ax.annotate(
                i, (gcentroid, 0.85), xycoords=('data', 'axes fraction'),
                color=color2, rotation=270, size='x-small',
                annotation_clip=True)
    return view


def load_galaxy_summary(galaxy, path):
    """
    CAUTION: This function alters its input.

    Param `galaxy` must be spectools GalaxySpec object
    """
    summary = read_summary(path)
    galaxy.objname = summary['galaxyname']
    galaxy.z = summary['redshift']
    if 'dataunit' in summary.keys():
        galaxy.dataunit = u.Unit(summary['dataunit'])
    if 'frequnit' in summary.keys():
        galaxy.frequnit = u.Unit(summary['frequnit'])
    if 'waveunit' in summary.keys():
        galaxy.waveunit = u.Unit(summary['waveunit'])
    if 'velunit' in summary.keys():
        galaxy.velunit = u.Unit(summary['velunit'])
    rts = []
    dep_transitions = []
    for line in summary['transitions'].keys():
        if "reference_transition" in summary["transitions"][line].keys():
            # For backward compatibility
            rt = summary["transitions"][line]["reference_transition"]
        else:
            # The modern way
            rt = summary["transitions"][line]["reference_transition_name"]
        if rt is not None:
            dep_transitions.append((line, rt))
            continue
        tmp = galaxy.add_transition(line)
        tmp.load_line_summary(summary['transitions'][line])
        rts.append(rt)

    for line, rt in dep_transitions:
        tmp = galaxy.add_transition(line, ref_transition=galaxy.transitions[rt])
        tmp.load_line_summary(summary["transitions"][line])


def load_summary(galspec, path):
    """
    Alias for load_galaxy_summary.
    Caution: This fundtion alters its input
    """
    load_galaxy_summary(galspec, path)


def read_summary(path):
    with open(path, 'r') as f:
        y = yaml.full_load(f)
    return y
