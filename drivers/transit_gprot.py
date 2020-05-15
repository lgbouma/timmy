"""
Fit data for transit + GP rotation model.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product

from timmy.modelfitter import ModelFitter, ModelParser
import timmy.plotting as tp
from timmy.convenience import get_clean_data
from timmy.priors import initialize_prior_d
from timmy.paths import RESULTSDIR

def main(modelid):

    make_threadsafe = 0

    traceplot = 0
    sampleplot = 1
    cornerplot = 1

    binsize = None # or 120*5

    OVERWRITE = 1
    REALID = 'TOI_837'
    provenance = 'spoc' # could be "cdips"
    yval = 'PDCSAP_FLUX' # could be SAP_FLUX 

    PLOTDIR = os.path.join(
        RESULTSDIR, '{}_phot_results'.format(REALID), '20200504'
    )

    ##########################################

    assert modelid == 'transit_gprot'

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

    x_obs, y_obs, y_err = get_clean_data(provenance, yval, binsize=binsize,
                                         maskflares=1)
    y_obs = (y_obs - 1)*1e3 # mean normalized and units: ppt.
    y_err *= 1e3

    mp = ModelParser(modelid)
    prior_d = initialize_prior_d(mp.modelcomponents)

    mstar, rstar = 1, 1 # actually about correct for 837
    m = ModelFitter(modelid, x_obs, y_obs, y_err, prior_d, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=OVERWRITE, mstar=mstar,
                    rstar=rstar)
    #FIXME from here
    #FIXME from here

    print(pm.summary(m.trace, varnames=list(prior_d.keys())))

    if make_threadsafe:
        pass

    else:
        if traceplot:
            outpath = join(PLOTDIR, '{}_{}_traceplot.png'.format(REALID, modelid))
            tp.plot_traceplot(m, outpath)

        if sampleplot:
            outpath = join(PLOTDIR, '{}_{}_sampleplot.png'.format(REALID, modelid))
            tp.plot_sampleplot(m, outpath, N_samples=10)

        if splitsignalplot:

            outpath = join(PLOTDIR, '{}_{}_splitsignalmap.png'.format(REALID, modelid))
            ydict = tp.plot_splitsignal_map(m, outpath)

            # outpath = join(PLOTDIR, '{}_{}_phasefoldmap.png'.format(REALID, modelid))
            # tp.plot_phasefold_map(m, ydict, outpath)

        if cornerplot:
            #prior_d.pop('omegaorb', None) # not sampled; only used in data generation
            #prior_d.pop('phiorb', None) # not sampled; only used in data generation
            outpath = join(PLOTDIR, '{}_{}_cornerplot.png'.format(REALID, modelid))
            tp.plot_cornerplot(prior_d, m, outpath)


if __name__ == "__main__":
    main('transit_gprot')
