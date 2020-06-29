"""
Fit single transit + quadratic trend (presumably a ground-based transit).

The prior should be chosen to be uninformative, if at all possible.
"""

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product
from copy import deepcopy

from timmy.modelfitter import ModelFitter, ModelParser
import timmy.plotting as tp
from timmy.convenience import (
    get_clean_tessphot, detrend_tessphot, get_elsauce_phot, _subset_cut
)
from timmy.priors import initialize_prior_d
from timmy.paths import RESULTSDIR

from collections import OrderedDict

def main(modelid):

    seldate = '20200426'
    bp = 'Rband'
    N_samples = 12000 # nb: 7.5mins, decent but not perfect convergence

    # seldate = '20200614'
    # bp = 'Bband'
    # Nsamples = 12000 # 7min: gets rhat = 1.0 for all parameters.

    cornerplot = 1
    phaseplot = 0
    grounddepth = 0
    fittedzoom = 0

    make_threadsafe = 0
    cut_tess = 1
    OVERWRITE = 1
    REALID = 'TOI_837'

    PLOTDIR = os.path.join(
        RESULTSDIR, f'{REALID}_{modelid}_{seldate}_{bp}_phot_results'
    )
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)
    PLOTDIR = os.path.join(PLOTDIR, f'20200617_{bp}')

    ##########################################

    assert modelid in ['onetransit']

    print(42*'#')
    print(modelid)
    print(42*'#')

    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'timmy',
        f'{REALID}_model_{modelid}_{seldate}_{bp}.pkl'
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

    if seldate not in datestrs:
        raise NotImplementedError
    else:
        usedata = OrderedDict()
        tra_ind = int(np.argwhere(np.array(datestrs) == seldate))
        usedata[f'elsauce_{tra_ind}'] = (
            deepcopy(datasets[f'elsauce_{tra_ind}'])
        )

    # note: we're fitting the detrended data
    mp = ModelParser(modelid)

    prior_d = initialize_prior_d(mp.modelcomponents, datasets=usedata)

    m = ModelFitter(modelid, usedata, prior_d, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=OVERWRITE, N_samples=N_samples)

    print(pm.summary(m.trace, var_names=list(prior_d.keys())))
    summdf = pm.summary(m.trace, var_names=list(prior_d.keys()), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    var_names = ['roughdepth']
    ddf = pm.summary(m.trace, var_names=var_names, round_to=10,
                     kind='stats', stat_funcs={'median':np.nanmedian},
                     extend=True)
    print(ddf)
    d_1sig = np.percentile(m.trace.roughdepth, 50-(68/2))
    d_2sig = np.percentile(m.trace.roughdepth, 50-(95/2))
    print(f'1 sigma depth lower limit: {d_1sig:.6f}')
    print(f'2 sigma depth lower limit: {d_2sig:.6f}')

    if make_threadsafe:
        pass

    else:
        if cornerplot:
            outpath = join(PLOTDIR, f'{REALID}_{modelid}_cornerplot.png')
            prior_d['roughdepth'] = 4600e-6
            tp.plot_cornerplot(prior_d, m, outpath)

if __name__ == "__main__":
    main('onetransit')
