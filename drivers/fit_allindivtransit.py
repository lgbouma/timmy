"""
Fit all transits (TESS+ground), allowing each transit a quadratic trend. No
"pre-detrending".
"""

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product

from timmy.modelfitter import ModelFitter, ModelParser
import timmy.plotting as tp
from timmy.convenience import (
    get_clean_tessphot, detrend_tessphot, get_elsauce_phot, _subset_cut,
    get_astep_phot
)
from timmy.priors import initialize_prior_d
from timmy.paths import RESULTSDIR

from collections import OrderedDict

from astrobase.lcmath import find_lc_timegroups

def main(modelid):

    make_threadsafe = 0
    cut_tess = 1

    fitindiv = 1
    phaseplot = 1
    grounddepth = 1
    cornerplot = 1
    subsetcorner = 1

    N_samples = 30000 # took 2h 20m, but Rhat=1.0 for all
    # N_samples = 2000 # took 16m, 14s. but Rhat=1.01 for b, rp/rs, + a few a1/a2s
    target_accept = 0.9

    OVERWRITE = 1
    REALID = 'TOI_837'

    PLOTDIR = os.path.join(
        RESULTSDIR, '{}_{}_phot_results'.format(REALID, modelid)
    )
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    datestr = '20200710'
    PLOTDIR = os.path.join(PLOTDIR, datestr)

    ##########################################

    assert modelid in ['allindivtransit']

    print(42*'#')
    print(modelid)
    print(42*'#')

    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'timmy',
        f'{REALID}_model_{modelid}_{datestr}.pkl'
    )
    np.random.seed(42)

    # get tess data
    provenance = 'spoc' # could be "cdips"
    yval = 'PDCSAP_FLUX' # could be SAP_FLUX 
    x_obs, y_obs, y_err = get_clean_tessphot(provenance, yval, binsize=None,
                                             maskflares=1)
    #y_flat, y_trend = detrend_tessphot(x_obs, y_obs, y_err)
    s = np.isfinite(y_obs) & np.isfinite(x_obs) & np.isfinite(y_err)
    x_obs, y_obs, y_err = x_obs[s], y_obs[s], y_err[s]
    if cut_tess:
        x_obs, y_obs, y_err = _subset_cut(x_obs, y_obs, y_err, n=3.5)

    ngroups, groupinds = find_lc_timegroups(x_obs, mingap=4.0)
    assert ngroups == 5

    datasets = OrderedDict()
    for ix, g in enumerate(groupinds):
        tess_texp = np.nanmedian(np.diff(x_obs[g]))
        datasets[f'tess_{ix}'] = [x_obs[g], y_obs[g], y_err[g], tess_texp]

    datestrs = ['20200401', '20200426', '20200521', '20200614']
    for ix, d in enumerate(datestrs):
        x_obs, y_obs, y_err = get_elsauce_phot(datestr=d)
        x_obs -= 2457000 # convert to BTJD
        elsauce_texp = np.nanmedian(np.diff(x_obs))
        datasets[f'elsauce_{ix}'] = [x_obs, y_obs, y_err, elsauce_texp]

    datestrs = ['20200529', '20200614', '20200623']
    for ix, d in enumerate(datestrs):
        x_obs, y_obs, y_err = get_astep_phot(datestr=d)
        x_obs += 2450000 # convert to BJD_TDB
        x_obs -= 2457000 # convert to BTJD
        astep_texp = np.nanmedian(np.diff(x_obs))
        datasets[f'astep_{ix}'] = [x_obs, y_obs, y_err, astep_texp]

    mp = ModelParser(modelid)

    prior_d = initialize_prior_d(mp.modelcomponents, datasets=datasets)

    m = ModelFitter(modelid, datasets, prior_d, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=OVERWRITE, N_samples=N_samples,
                    target_accept=target_accept)

    print(pm.summary(m.trace, var_names=list(prior_d.keys())))
    summdf = pm.summary(m.trace, var_names=list(prior_d.keys()), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)
    rp_limit = np.percentile(m.trace.r_planet, 1-0.9973)
    print(42*'-')
    print(f'Rp limit: {rp_limit:.3f} Rjup')
    print(42*'-')

    if make_threadsafe:
        pass

    else:
        if grounddepth:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_grounddepth.png')
            tp.plot_grounddepth(m, summdf, outpath, modelid=modelid,
                                showerror=0)

        if subsetcorner:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_subsetcorner.png')
            tp.plot_subsetcorner(m, outpath)

        if phaseplot:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_phaseplot.png')
            tp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1)

        if fitindiv:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_fitindiv.png')
            tp.plot_fitindiv(m, summdf, outpath, modelid=modelid)

        if cornerplot:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_cornerplot.png')
            tp.plot_cornerplot(prior_d, m, outpath)


if __name__ == "__main__":
    main('allindivtransit')
