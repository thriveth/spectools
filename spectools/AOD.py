#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import lmfit as lm
import numpy as np
from astropy.table import Table

def aod(flam, logN_ion, f_c):
    """ Returns flux as function of f*lambda, column density and covering
    fraction.
    """
    N_ion = 10**logN_ion
    tau = flam * N_ion / 3.768e14
    rel_flux = 1 - f_c * (1 - np.exp(-tau))
    return rel_flux


def fit_single_bin(flams, fluxes, errs, mc=True, method="nelder"):
    """ Fluxes must be array
    """
    model = lm.Model(aod)
    pars = model.make_params()
    pars['logN_ion'].set(13, min=8, max=16)
    pars['f_c'].set(0.5, min=0, max=1)
    if not mc:
        pars['logN_ion'].set(min=10, max=14)
    result = model.fit(
        fluxes, params=pars, flam=flams, weights=1/errs, method=method)
    result.fit(method="leastsq")
    MLE = result.best_values
    if mc:
        mcresult = result.emcee()#burn=100)
        MLE = mcresult.params.valuesdict()#best_values
        #MLElogN = mcresult.flatchain['logN_ion'].argmax()
        return MLE, mcresult
    else:
        return MLE, result

def brute_fit_bin(flams, fluxes, errs):
    covfrac = np.linspace(0., 1., 51)
    logN = np.linspace(10., 20., 51)
    chisqarr = np.ones((len(covfrac), len(logN)))
    CF, NC = np.meshgrid(covfrac, logN)
    Flam = flams.reshape(1, 1, -1)
    #Wav = vels[mask].reshape(1, 1, -1)
    CF = np.dstack([CF] * Flam.shape[2])
    NC = np.dstack([NC] * Flam.shape[2])
    #NC = 10. ** NC
    # tau = -1. * Flam * NC / 3.768e14
    # factorx = 1. - np.exp(tau)
    # rlflx = 1. - CF * (1. - np.exp(tau))
    rlflx = aod(Flam, NC, CF)
    resid = (rlflx - fluxes)**2 / errs**2
    chisq = (resid ** 2).sum(axis=2)# / len(fluxes)
    # Now: best fit and marginalization
    minidx = np.where(chisq == chisq.min())
    confidx = np.where(chisq < chisq.min() + 1)
    fitlogN = [
        logN[minidx[0]][0],
        logN[confidx[0]].max(),
        logN[confidx[0]].min()]
    fitcf = [
        covfrac[minidx[1]][0],
        covfrac[confidx[1]].max(),
        covfrac[confidx[1]].min()]
    return chisq, {'logN_ion': fitlogN, 'f_c': fitcf}

def postprocess_fit(fitresult, mode):
    """ Placeholder and reminder, I need a function to do this"""
    # TODO: Implement!
    return


def prepare_fit_range(indata, bounds=None):
    """ `indata` must be binned on common velocity grid

    If no range is passed, full data range is fitted.
    """
    mcols = {}
    if not bounds:
        bounds = indata['wave'].min(), indata['wave'].max()
    bounds = np.array(bounds)
    idx = np.where(
        (indata['Velocity'] > bounds.min())
        & (indata['Velocity'] < bounds.max())
    )
    fitdata = indata[idx]
    mcols["Velocity"] = fitdata["Velocity"]
    for col in fitdata.colnames:
        if not col.endswith('flux'):# & col.startswith("Si"):
            continue
        if not col[:-5] in fikdict:
            continue
        species = " ".join(col.split(" ")[:-1])
        mflcol = Table.MaskedColumn(
            fitdata[col], name="col", mask=fitdata[species+" mask"].data)
        mercol = Table.MaskedColumn(
            fitdata[species+" errs"], mask=fitdata[species+" mask"].data)
        mcols[col] = mflcol
        mcols[species+" errs"] = mercol
    fittable = Table(mcols)
    return fittable


def fit_range(intable, bounds=None, mc=True, verbose=False, fitmethod='lmfit',
              quantiles=(0.16, 0.84)):
    """Takes data astropy.Table in vflis.ecsv format. Fits  AOD stuff."""
    # Cut out desired ranges
    fittable = prepare_fit_range(intable, bounds)
    # Table of fluxes only
    fluxcols = [col for col in fittable.colnames if col.endswith("flux")]
    # Table of errors only
    errscols = [col for col in fittable.colnames if col.endswith("errs")]
    # List of transitions (not species, actually)
    species = [
        " ".join(col.split()[:-1]) for col in fittable.colnames
        if col.endswith("flux")
    ]
    # Now, make fixed-format arrays to pass to fitter
    fluxarr = np.array([fittable[a] for a in fluxcols]).T
    errsarr = np.array([fittable[e] for e in errscols]).T
    maskarr = np.array([fittable[m].mask for m in fluxcols]).T
    flams = np.array([fikdict[s] * wlsdict[s] for s in species if s in fikdict])
    # Now, loop through all the velocity bins
    reslines = []  # Temporary container for output until postprocessing
    fitresults = []
    for i in range(fluxarr.shape[0]):
        vel = fittable["Velocity"][i]
        idx = np.where(np.invert(maskarr[i, :]))[0]  # Don't use masked points
        print("\r", i+1,"/", fluxarr.shape[0], ", Datapoints: ", len(idx), end="")
        if len(idx) < 2:
            reslines.append([vel, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            fitresults.append(None)
            continue
        flam = flams[idx]
        flux = fluxarr[i, idx]
        errs = errsarr[i, idx]
        if fitmethod == 'lmfit':
            MLE, result = fit_single_bin(flam, flux, errs, mc=mc)
            if mc:
                flower, fupper = \
                    result.flatchain['f_c'].quantile(quantiles[0]), \
                    result.flatchain['f_c'].quantile(quantiles[1])
                Nlower, Nupper = \
                    result.flatchain['logN_ion'].quantile(quantiles[0]), \
                    result.flatchain['logN_ion'].quantile(quantiles[1])
            else:
                flower, fupper = \
                    result.params['f_c'].value-result.params['f_c'].stderr,\
                    result.params['f_c'].value+result.params['f_c'].stderr,
                Nlower, Nupper = \
                    result.params['logN_ion'].value - result.params['logN_ion'].stderr,\
                    result.params['logN_ion'].value + result.params['logN_ion'].stderr,
            resline = [
                vel,
                MLE['f_c'], flower, fupper,#.value,
                MLE['logN_ion'], Nlower, Nupper#.value,
            ]
        elif fitmethod == 'brute':
            result, outpars = brute_fit_bin(flam, flux, errs)
            #resline = [
                #vel,
                #outpars['f_c'], outpars['f_c err'],
                #outpars['N_ion'], outpars['N_ion']]
            resline = [vel] + outpars["f_c"] + outpars["logN_ion"]
        reslines.append(resline)
        fitresults.append(result)
    reslines = np.array(reslines)
    restable = Table(
        reslines,
        names=["Velocity", "f_c", "f_c lower", "f_c upper",
               "log N", "log N lower", "log N upper"]
    )
    if verbose:
        restable["Fitters"] = fitresults
    return restable


def show_phase(intable, bounds=None):
    # TODO Implement the rest!
    showtable = prepare_fit_range(intable, bounds)
    return showtable


fikdict = {
    'Si II 1190': 0.277,
    'Si II 1193': 0.575,
    'Si II 1260': 1.22,
    'Si II 1304': 0.0928,
    'Si IV 1122': 0.807,
    'Si IV 1393': 0.513,
    'Si IV 1402': 0.255,
    'Si II 1526': 0.133
}

wlsdict = {
    'Si IV 1122': 1122.4849,
    'Si II 1190': 1190.4158,
    'Si II 1193': 1193.2897,
    'Si III 1206': 1206.4995,
    'Si II 1260': 1260.4221,
    'O I 1302': 1302.16848,
    'Si II 1304': 1304.3702,
    'C II 1334': 1334.5323,
    'Si IV 1393': 1393.7546,
    'Si IV 1402': 1402.7697,
    'Si II 1526': 1526.7066
}
