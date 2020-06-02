"""
Fit all transits (TESS+ground), after "detrending" the stellar variability.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product

from timmy.modelfitter import ModelFitter, ModelParser
import timmy.plotting as tp
from timmy.convenience import (
    get_clean_tessphot, detrend_tessphot, get_elsauce_phot
)
from timmy.priors import initialize_prior_d
from timmy.paths import RESULTSDIR

from collections import OrderedDict

def main(modelid):

    make_threadsafe = 0
    phaseplot = 1
    cornerplot = 1
    fittedzoom = 1

    traceplot = 0
    sampleplot = 0
    splitsignalplot = 0

    OVERWRITE = 1
    REALID = 'TOI_837'

    PLOTDIR = os.path.join(
        RESULTSDIR, '{}_{}_phot_results'.format(REALID, modelid)
    )
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    PLOTDIR = os.path.join(PLOTDIR, '20200518')

    ##########################################

    assert modelid == 'transit'

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
    tess_texp = np.nanmedian(np.diff(x_obs))

    datasets = OrderedDict()
    datasets['tess'] = [x_obs, y_flat, y_err, tess_texp]

    x_obs, y_obs, y_err = get_elsauce_phot()
    elsauce_texp = np.nanmedian(np.diff(x_obs))
    datasets['elsauce'] = [x_obs, y_obs, y_err, elsauce_texp]

    # note: we're fitting the detrended data
    mp = ModelParser(modelid)
    prior_d = initialize_prior_d(mp.modelcomponents)

    # NOTE: will need to if/else the datasets assignment in lieu of a
    # dataframe.
    m = ModelFitter(modelid, datasets, prior_d, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=OVERWRITE)

    print(pm.summary(m.trace, var_names=list(prior_d.keys())))
    summdf = pm.summary(m.trace, var_names=list(prior_d.keys()), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    if make_threadsafe:
        pass

    else:
        if fittedzoom:
            outpath = join(PLOTDIR, '{}_{}_fittedzoom.png'.format(REALID, modelid))
            tp.plot_fitted_zoom(m, summdf, outpath)

        if phaseplot:
            outpath = join(PLOTDIR, '{}_{}_phaseplot.png'.format(REALID, modelid))
            tp.plot_phasefold(m, summdf, outpath)

        if cornerplot:
            outpath = join(PLOTDIR, '{}_{}_cornerplot.png'.format(REALID, modelid))
            tp.plot_cornerplot(prior_d, m, outpath)

        # NOTE: following are deprecated
        if sampleplot:
            outpath = join(PLOTDIR, '{}_{}_sampleplot.png'.format(REALID, modelid))
            tp.plot_sampleplot(m, outpath, N_samples=100)



if __name__ == "__main__":
    main('multitransit')
