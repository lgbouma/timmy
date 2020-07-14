import os, re
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pymc3 as pm
from os.path import join
from itertools import product

from timmy.modelfitter import ModelFitter, ModelParser
from timmy.priors import initialize_prior_d
from timmy.paths import RESULTSDIR
from timmy.convenience import (
    get_clean_tessphot, get_elsauce_phot, _subset_cut, get_astep_phot
)
from collections import OrderedDict
from astrobase.lcmath import find_lc_timegroups

def main(modelid, datestr):

    assert modelid in ['allindivtransit', 'tessindivtransit']
    yval = 'PDCSAP_FLUX' # could be SAP_FLUX 

    OVERWRITE = 0
    REALID = 'TOI_837'
    provenance = 'spoc'
    PLOTDIR = os.path.join(
        RESULTSDIR, '{}_{}_phot_results'.format(REALID, modelid)
    )
    PLOTDIR = os.path.join(PLOTDIR, datestr)

    summarypath = os.path.join(
        PLOTDIR, 'posterior_table_raw_{}.csv'.format(modelid)
    )
    pklpath = os.path.join(
        os.path.expanduser('~'), 'local', 'timmy',
        '{}_model_{}_{}.pkl'.format(REALID, modelid, datestr)
    )
    np.random.seed(42)

    ########################################## 
    # get allindivtransit initialized
    ########################################## 
    provenance = 'spoc' # could be "cdips"
    yval = 'PDCSAP_FLUX' # could be SAP_FLUX 
    x_obs, y_obs, y_err = get_clean_tessphot(provenance, yval, binsize=None,
                                             maskflares=1)
    s = np.isfinite(y_obs) & np.isfinite(x_obs) & np.isfinite(y_err)
    x_obs, y_obs, y_err = x_obs[s], y_obs[s], y_err[s]
    cut_tess = 1
    if cut_tess:
        x_obs, y_obs, y_err = _subset_cut(x_obs, y_obs, y_err, n=3.5)

    ngroups, groupinds = find_lc_timegroups(x_obs, mingap=4.0)
    assert ngroups == 5

    datasets = OrderedDict()
    for ix, g in enumerate(groupinds):
        tess_texp = np.nanmedian(np.diff(x_obs[g]))
        datasets[f'tess_{ix}'] = [x_obs[g], y_obs[g], y_err[g], tess_texp]

    if modelid == 'allindivtransit':
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
    ########################################## 
    # end intiialization
    ########################################## 

    if not os.path.exists(summarypath):

        m = ModelFitter(modelid, datasets, prior_d, plotdir=PLOTDIR,
                        pklpath=pklpath, overwrite=OVERWRITE)

        # stat_funcsdict = A list of functions or a dict of functions with
        # function names as keys used to calculate statistics. By default, the
        # mean, standard deviation, simulation standard error, and highest
        # posterior density intervals are included.
        stat_funcsdict = {
            'median': np.nanmedian
        }

        df = pm.summary(
            m.trace,
            round_to=10, kind='stats',
            stat_funcs=stat_funcsdict,
            extend=True
        )

        df.to_csv(summarypath, index=True)

    else:
        df = pd.read_csv(summarypath, index_col=0)

    fitted_params = [
        'period', 't0', 'log_r', 'b', 'u[0]', 'u[1]', 'r_star', 'logg_star'
    ]
    for i in range(5):
        fitted_params.append(f'tess_{i}_mean')
        fitted_params.append(f'tess_{i}_a1')
        fitted_params.append(f'tess_{i}_a2')
    if modelid == 'allindivtransit':
        for i in range(4):
            fitted_params.append(f'elsauce_{i}_mean')
            fitted_params.append(f'elsauce_{i}_a1')
            fitted_params.append(f'elsauce_{i}_a2')
        for i in range(3):
            fitted_params.append(f'astep_{i}_mean')
            fitted_params.append(f'astep_{i}_a1')
            fitted_params.append(f'astep_{i}_a2')

    n_fitted = len(fitted_params)

    derived_params = [
        'r', 'rho_star', 'r_planet', 'a_Rs', 'cosi', 'T_14', 'T_13'
    ]
    n_derived = len(derived_params)

    srows = []
    for f in fitted_params:
        srows.append(f)
    for d in derived_params:
        srows.append(d)

    df = df.loc[srows]

    cols = ['median', 'mean', 'sd', 'hpd_3%', 'hpd_97%']

    df = df[cols]

    print(df)

    from timmy.priors import (
        LOGG, LOGG_STDEV, RSTAR, RSTAR_STDEV
    )

    delta_u = 0.15
    pr = {
        'period': normal_str(
            mu=prior_d['period'], sd=1e-1, fmtstr='({:.4f}; {:.4f})'
        ),
        't0': normal_str(
            mu=prior_d['t0'], sd=1e-1, fmtstr='({:.6f}; {:.4f})'
        ),
        'log_r': uniform_str(
            lower=np.log(1e-2), upper=np.log(1), fmtstr='({:.3f}; {:.3f})'
        ),
        'b': r'$\mathcal{U}(0; 1+R_{\mathrm{p}}/R_\star)$',
        #'u[0]': '(2)',
        #'u[1]': '(2)',
        'u[0]': uniform_str(prior_d['u[0]']-delta_u, prior_d['u[0]']+delta_u,
                            fmtstr='({:.3f}; {:.3f})') + '$^{(2)}$',
        'u[1]': uniform_str(prior_d['u[1]']-delta_u, prior_d['u[1]']+delta_u,
                            fmtstr='({:.3f}; {:.3f})') + '$^{(2)}$',
        'r_star': truncnormal_str(
            mu=RSTAR, sd=RSTAR_STDEV, fmtstr='({:.3f}; {:.3f})'
        ),
        'logg_star': normal_str(
            mu=LOGG, sd=LOGG_STDEV, fmtstr='({:.3f}; {:.3f})'
        )
    }
    ufmt = '({:.2f}; {:.2f})'

    delta_trend = 0.05
    for i in range(5):
        pr[f'tess_{i}_mean'] = normal_str(mu=prior_d[f'tess_{i}_mean'],
                                          sd=0.01, fmtstr=ufmt)
        pr[f'tess_{i}_a1'] = uniform_str(lower=-delta_trend, upper=delta_trend, fmtstr=ufmt)
        pr[f'tess_{i}_a2'] = uniform_str(lower=-delta_trend, upper=delta_trend, fmtstr=ufmt)
    if modelid == 'allindivtransit':
        for i in range(4):
            pr[f'elsauce_{i}_mean'] = normal_str(mu=prior_d[f'elsauce_{i}_mean'],
                                                 sd=0.01, fmtstr=ufmt)
            pr[f'elsauce_{i}_a1'] = uniform_str(lower=-delta_trend, upper=delta_trend, fmtstr=ufmt)
            pr[f'elsauce_{i}_a2'] = uniform_str(lower=-delta_trend, upper=delta_trend, fmtstr=ufmt)
        for i in range(3):
            pr[f'astep_{i}_mean'] = normal_str(mu=prior_d[f'astep_{i}_mean'],
                                                 sd=0.01, fmtstr=ufmt)
            pr[f'astep_{i}_a1'] = uniform_str(lower=-delta_trend, upper=delta_trend, fmtstr=ufmt)
            pr[f'astep_{i}_a2'] = uniform_str(lower=-delta_trend, upper=delta_trend, fmtstr=ufmt)


    for d in derived_params:
        pr[d] = '--'

    # round everything. requires a double transpose because df.round
    # operates column-wise
    if modelid in ['allindivtransit', 'tessindivtransit']:
        round_precision = [7, 7, 5, 4, 3, 3, 3, 3]
        n_rp = len(round_precision)
        for i in range(n_fitted - n_rp):
            round_precision.append(4)
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
        'log_r': '--',
        'b': '--',
        'u[0]': '--',
        'u[1]': '--',
        'r_star': r'$R_\odot$',
        'logg_star': 'cgs'
    }
    for i in range(5):
        ud[f'tess_{i}_mean'] = '--'
        ud[f'tess_{i}_a1'] = 'd$^{-1}$'
        ud[f'tess_{i}_a2'] = 'd$^{-2}$'
    if modelid == 'allindivtransit':
        for i in range(4):
            ud[f'elsauce_{i}_mean'] = '--'
            ud[f'elsauce_{i}_a1'] = 'd$^{-1}$'
            ud[f'elsauce_{i}_a2'] = 'd$^{-2}$'
        for i in range(3):
            ud[f'astep_{i}_mean'] = '--'
            ud[f'astep_{i}_a1'] = 'd$^{-1}$'
            ud[f'astep_{i}_a2'] = 'd$^{-2}$'

    ud['r'] = '--'
    ud['rho_star'] = 'g$\ $cm$^{-3}$'
    ud['r_planet'] = '$R_{\mathrm{Jup}}$'
    ud['a_Rs'] = '--'
    ud['cosi'] = '--'
    ud['T_14'] = 'hr'
    ud['T_13'] = 'hr'

    df['units'] = list(ud.values())

    df = df[
        ['units', 'priors', 'median', 'mean', 'sd', 'hpd_3%', 'hpd_97%']
    ]

    latexparams = [
        #useful
        r"$P$",
        r"$t_0^{(1)}$",
        r"$\log R_{\rm p}/R_\star$",
        "$b$",
        "$u_1$",
        "$u_2$",
        "$R_\star$",
        "$\log g$"
    ]
    for i in range(5):
        latexparams.append('$a_{'+str(i)+'0;\mathrm{TESS}}$')
        latexparams.append('$a_{'+str(i)+'1;\mathrm{TESS}}$')
        latexparams.append('$a_{'+str(i)+'2;\mathrm{TESS}}$')
    if modelid == 'allindivtransit':
        for i in range(4):
            latexparams.append('$a_{'+str(i)+'0;\mathrm{Sauce}}$')
            latexparams.append('$a_{'+str(i)+'1;\mathrm{Sauce}}$')
            latexparams.append('$a_{'+str(i)+'2;\mathrm{Sauce}}$')
        for i in range(3):
            latexparams.append('$a_{'+str(i)+'0;\mathrm{ASTEP}}$')
            latexparams.append('$a_{'+str(i)+'1;\mathrm{ASTEP}}$')
            latexparams.append('$a_{'+str(i)+'2;\mathrm{ASTEP}}$')

    from billy.convenience import flatten
    dlatexparams = [
        r"$R_{\rm p}/R_\star$",
        r"$\rho_\star$",
        r"$R_{\rm p}$",
        "$a/R_\star$",
        '$\cos i$',
        '$T_{14}$',
        '$T_{13}$'
    ]
    latexparams = flatten([latexparams, dlatexparams])
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


def lognormal_str(mu, sd, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\log\\mathcal{N}'+'({}; {})$'.format(mu, sd)
    else:
        return '$\log\\mathcal{N}'+'{}$'.format(fmtstr).format(mu, sd)


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


def loguniform_str(lower, upper, fmtstr=None):
    # fmtstr: e.g., "({:.5f}; {:.2f})"
    if fmtstr is None:
        return '$\log\\mathcal{U}'+'({}; {})$'.format(lower, upper)
    else:
        return '$\log\\mathcal{U}'+'{}$'.format(fmtstr).format(lower, upper)



if __name__ == "__main__":

    main('tessindivtransit', '20200711')
    main('allindivtransit', '20200711')
