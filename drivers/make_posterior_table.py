import os, re
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product

from timmy.modelfitter import ModelFitter, ModelParser
from timmy.convenience import get_clean_data, detrend_data
from timmy.priors import initialize_prior_d
from timmy.paths import RESULTSDIR

def main(modelid):

    assert modelid == 'transit'
    yval = 'PDCSAP_FLUX' # could be SAP_FLUX 

    OVERWRITE = 0
    REALID = 'TOI_837'
    provenance = 'spoc'
    PLOTDIR = os.path.join(
        RESULTSDIR, '{}_{}_phot_results'.format(REALID, modelid)
    )
    PLOTDIR = os.path.join(PLOTDIR, '20200515')

    summarypath = os.path.join(
        PLOTDIR, 'posterior_table_raw_{}.csv'.format(modelid)
    )
    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'timmy',
        '{}_model_{}.pkl'.format(REALID, modelid)
    )
    np.random.seed(42)

    x_obs, y_obs, y_err = get_clean_data(provenance, yval, binsize=None,
                                         maskflares=1)
    y_flat, y_trend = detrend_data(x_obs, y_obs, y_err)
    s = np.isfinite(y_flat) & np.isfinite(x_obs) & np.isfinite(y_err)
    x_obs, y_flat, y_err = x_obs[s], y_flat[s], y_err[s]

    # note: we're fitting the detrended data
    mp = ModelParser(modelid)
    prior_d = initialize_prior_d(mp.modelcomponents)

    if not os.path.exists(summarypath):

        m = ModelFitter(modelid, x_obs, y_flat, y_err, prior_d, plotdir=PLOTDIR,
                        pklpath=pklpath, overwrite=OVERWRITE)

        df = pm.summary(
            m.trace,
            round_to=10, kind='stats'
        )

        df.to_csv(summarypath, index=True)

    else:
        df = pd.read_csv(summarypath, index_col=0)

    fitted_params = [
        'period', 't0', 'r', 'b', 'u[0]', 'u[1]', 'mean', 'r_star', 'logg_star'
    ]
    derived_params = [
        'rho_star', 'r_planet', 'a_Rs', 'cosi', 'T_14', 'T_13'
    ]

    srows = []
    for f in fitted_params:
        srows.append(f)
    for d in derived_params:
        srows.append(d)

    df = df.loc[srows]

    print(df)

    from timmy.priors import (
        LOGG, LOGG_STDEV, RSTAR, RSTAR_STDEV
    )

    pr = {
        'period': normal_str(
            mu=prior_d['period'], sd=5e-3, fmtstr='({:.4f}; {:.4f})'
        ),
        't0': normal_str(
            mu=prior_d['t0'], sd=5e-3, fmtstr='({:.6f}; {:.4f})'
        ),
        'r': r'$\mathcal{U}(10^{-3}; 1)$',
        'b': r'$\mathcal{U}(0; 1+R_{\mathrm{p}}/R_\star)$',
        'u[0]': '(2)',
        'u[1]': '(2)',
        'mean': uniform_str(
            lower=prior_d['mean']-1e-2, upper=prior_d['mean']+1e-2
        ),
        'r_star': truncnormal_str(
            mu=RSTAR, sd=RSTAR_STDEV, fmtstr='({:.2f}; {:.2f})'
        ),
        'logg_star': normal_str(
            mu=LOGG, sd=LOGG_STDEV, fmtstr='({:.2f}; {:.2f})'
        )
    }
    for d in derived_params:
        pr[d] = '--'

    # round everything. requires a double transpose because df.round
    # operates column-wise
    if modelid == 'transit':
        round_precision = [7, 7, 5, 4, 3, 3, 6, 2, 2]
    else:
        raise NotImplementedError
    for d in derived_params:
        round_precision.append(2)

    df = df.T.round(
        decimals=dict(
            zip(df.index, round_precision)
        )
    ).T

    df['priors'] = list(pr.values())

    # units
    ud = {
        'period': 'd',
        't0': 'd',
        'r': '--',
        'b': '--',
        'u[0]': '--',
        'u[1]': '--',
        'mean': '--',
        'r_star': r'$R_\odot$',
        'logg_star': 'cgs'
    }

    ud['rho_star'] = 'g$\ $cm$^{-3}$'
    ud['r_planet'] = '$R_{\mathrm{Jup}}$'
    ud['a_Rs'] = '--'
    ud['cosi'] = '--'
    ud['T_14'] = 'hr'
    ud['T_13'] = 'hr'

    df['units'] = list(ud.values())

    df = df[
        ['units', 'priors', 'mean', 'sd', 'hpd_3%', 'hpd_97%']
    ]

    latexparams = [
        r"$P_{\rm s}$",
        r"$t_{\rm s}^{(1)}$",
        r"$R_{\rm p}/R_\star$",
        "$b$",
        "$u_1$",
        "$u_2$",
        "Mean",
        "$R_\star$",
        "$\log g$",
        # derived
        r"$\rho_\star$",
        r"$R_{\rm p}$",
        "$a/R_\star$",
        '$\cos i$',
        '$T_{14}$',
        '$T_{13}$'
    ]
    df.index = latexparams

    outpath = os.path.join(PLOTDIR,
                           'posterior_table_clean_{}.csv'.format(modelid))
    df.to_csv(outpath, float_format='%.12f', na_rep='NaN')
    print('made {}'.format(outpath))

    # df.to_latex is dumb with float formatting.
    outpath = os.path.join(PLOTDIR,
                           'posterior_table_clean_{}.tex'.format(modelid))
    df.to_csv(outpath, sep=',', line_terminator=' \\\\\n',
              float_format='%.12f', na_rep='NaN')

    with open(outpath, 'r') as f:
        lines = f.readlines()

    for ix, l in enumerate(lines):

        # replace commas with latex ampersands
        thisline = deepcopy(l.replace(',', ' & '))

        # replace quotes with nada
        thisline = thisline.replace('"', '')

        # replace }0 with },0
        thisline = thisline.replace('}0', '},0')
        thisline = thisline.replace('}1', '},1')
        thisline = thisline.replace('}2', '},2')

        if ix == 0:
            lines[ix] = thisline
            continue

        # iteratively replace stupid trailing zeros with whitespace
        while re.search("0{2,10}\ ", thisline) is not None:
            r = re.search("0{2,10}\ ", thisline)
            thisline = thisline.replace(
                thisline[r.start():r.end()],
                ' '
            )

        lines[ix] = thisline

    with open(outpath, 'w') as f:
        f.writelines(lines)

    print('made {}'.format(outpath))


def normal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{N}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\\mathcal{N}'+'{}$'.format(fmtstr).format(mu, sd)


def truncnormal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{T}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\\mathcal{T}'+'{}$'.format(fmtstr).format(mu, sd)


def uniform_str(lower, upper, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\\mathcal{U}'+'({}; {})$'.format(lower, upper)
    else:
        return '$\\mathcal{U}'+'{}$'.format(fmtstr).format(lower, upper)


if __name__ == "__main__":

    main('transit')
