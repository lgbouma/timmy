"""
Fit data for transit alone, after "detrending" the stellar variability.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join

from timmy.modelfitter import ModelFitter, ModelParser
import timmy.plotting as tp
from timmy.convenience import get_rv_data

from timmy.paths import RESULTSDIR

def main(modelid):

    assert modelid == 'rv'

    make_threadsafe = 0

    phaseplot = 0
    cornerplot = 0
    fittedzoom = 1

    OVERWRITE = 1
    REALID = 'TOI_837'

    PLOTDIR = os.path.join(
        RESULTSDIR, '{}_{}_fitting_results'.format(REALID, modelid)
    )
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    PLOTDIR = os.path.join(PLOTDIR, '20200525')
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    print(42*'#')
    print(modelid)
    print(42*'#')

    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'timmy',
        '{}_model_{}.pkl'.format(REALID, modelid)
    )
    np.random.seed(42)

    rv_df = get_rv_data()

    mp = ModelParser(modelid)

    prior_d = initialize_prior_d(mp.modelcomponents)

    m = ModelFitter(
        modelid, rv_df, prior_d, plotdir=PLOTDIR, pklpath=pklpath,
        overwrite=OVERWRITE
    )

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
        if traceplot:
            outpath = join(PLOTDIR, '{}_{}_traceplot.png'.format(REALID, modelid))
            tp.plot_traceplot(m, outpath)

        if sampleplot:
            outpath = join(PLOTDIR, '{}_{}_sampleplot.png'.format(REALID, modelid))
            tp.plot_sampleplot(m, outpath, N_samples=100)

        if splitsignalplot:
            outpath = join(PLOTDIR, '{}_{}_splitsignalmap.png'.format(REALID, modelid))
            ydict = tp.plot_splitsignal_map(m, outpath)


if __name__ == "__main__":
    main('rv')
