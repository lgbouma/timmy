"""
Fit all transits (TESS+ground), after "detrending" the stellar variability.

modelids implemented are "alltransit" and "alltransit_quad".

The former is a model in which there is a single set of shared planet
parameters across all instruments; only the means between instruments changes,
to account for the ground-based data.

The latter adds quadratic trends to the ground-based data, to enable more
careful assessment of the depth. (And to account for the changing airmass).
"""

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product

from timmy.modelfitter import ModelFitter, ModelParser
import timmy.plotting as tp
from timmy.convenience import (
    get_clean_tessphot, detrend_tessphot, get_elsauce_phot, _subset_cut
)
from timmy.priors import initialize_prior_d
from timmy.paths import RESULTSDIR

from collections import OrderedDict

def main(modelid):

    make_threadsafe = 0
    cut_tess = 1

    phaseplot = 0
    fittedzoom = 0
    grounddepth = 1
    cornerplot = 0

    OVERWRITE = 1
    REALID = 'TOI_837'

    PLOTDIR = os.path.join(
        RESULTSDIR, '{}_{}_phot_results'.format(REALID, modelid)
    )
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    PLOTDIR = os.path.join(PLOTDIR, '20200617')

    ##########################################

    assert modelid == 'alltransit' or modelid == 'alltransit_quad'

    print(42*'#')
    print(modelid)
    print(42*'#')

    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'timmy',
        '{}_model_{}.pkl'.format(REALID, modelid)
    )
    np.random.seed(42)

    # get tess data
    provenance = 'spoc' # could be "cdips"
    yval = 'PDCSAP_FLUX' # could be SAP_FLUX 
    x_obs, y_obs, y_err = get_clean_tessphot(provenance, yval, binsize=None,
                                             maskflares=1)
    y_flat, y_trend = detrend_tessphot(x_obs, y_obs, y_err)
    s = np.isfinite(y_flat) & np.isfinite(x_obs) & np.isfinite(y_err)
    x_obs, y_flat, y_err = x_obs[s], y_flat[s], y_err[s]
    if cut_tess:
        x_obs, y_flat, y_err = _subset_cut(x_obs, y_flat, y_err)
    tess_texp = np.nanmedian(np.diff(x_obs))

    datasets = OrderedDict()
    datasets['tess'] = [x_obs, y_flat, y_err, tess_texp]

    datestrs = ['20200401', '20200426', '20200521', '20200614']
    for ix, d in enumerate(datestrs):
        x_obs, y_obs, y_err = get_elsauce_phot(datestr=d)
        x_obs -= 2457000 # convert to BTJD
        elsauce_texp = np.nanmedian(np.diff(x_obs))
        datasets[f'elsauce_{ix}'] = [x_obs, y_obs, y_err, elsauce_texp]

    # note: we're fitting the detrended data
    mp = ModelParser(modelid)

    prior_d = initialize_prior_d(mp.modelcomponents, datasets=datasets)

    m = ModelFitter(modelid, datasets, prior_d, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=OVERWRITE)

    print(pm.summary(m.trace, var_names=list(prior_d.keys())))
    summdf = pm.summary(m.trace, var_names=list(prior_d.keys()), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    if make_threadsafe:
        pass

    else:
        if cornerplot:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_cornerplot.png')
            tp.plot_cornerplot(prior_d, m, outpath)

        if phaseplot:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_phaseplot.png')
            tp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1)

        if fittedzoom:
            outpath = join(PLOTDIR, '{}_{}_fittedzoom.png'.format(REALID, modelid))
            tp.plot_fitted_zoom(m, summdf, outpath, modelid=modelid)

        if grounddepth:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_grounddepth.png')
            tp.plot_grounddepth(m, summdf, outpath, modelid=modelid)



if __name__ == "__main__":
    # main('alltransit')
    main('alltransit_quad')
