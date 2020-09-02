"""
To get a more conservative limit on K as a function of P:
    1. fit a linear relation to the data, using a uniform (non-logarithmic
    prior) on the slope, that allows for positive and negative values.
    2. The slope distribution will be tightly constrained, giving you a robust 3
    sigma upper limit on its absolute value.
    3. Then you can ask for a given value of P, what value of K would yield a
    first derivative greater than the slope limit more than 99.7% of the time.
"""

import pymc3 as pm
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pickle, os, corner
from pymc3.backends.tracetab import trace_to_dataframe
from numpy import array as nparr
from astropy import units as u, constants as const

from datetime import datetime

use_exoplanet = 0

if use_exoplanet:
    from exoplanet.orbits import KeplerianOrbit

from numpy import polyfit
import multiprocessing as mp
from timmy.priors import RSTAR, MSTAR

from timmy.paths import DATADIR, RESULTSDIR
rdir = os.path.join(RESULTSDIR, 'rvlimits_method2')

def get_K_lim(period=100*u.day, gammadot_limit=0.82*u.m/u.s/u.day, frac=0.997):
    """
    Given a period and limiting dv/dt,
    return the minimum K such that

        |K ω sin(ωt)| < gammadot_limit

    at least frac % of the time. (Frac is expressed where 99.7% = 0.997)
    """
    assert frac == 0.997 # we use an approximation.

    ω = 2*np.pi/(period.to(u.day).value)
    K = (gammadot_limit.to(u.m/u.s/u.day).value) / ω

    return K


def rv_injection_worker(task):

    logK, logP, t0, sini, t_observed, gammadot_limit = task
    ecc = 0

    if use_exoplanet:
        # slow, and bringing a machine-gun to a knife-fight
        orbit = KeplerianOrbit(
            period=np.exp(logP), b=0, t0=t0, r_star=RSTAR, m_star=MSTAR
        )
        rv = orbit.get_radial_velocity(t_observed, K=np.exp(logK),
                                       output_units=u.m/u.s)
        _rv = rv.eval()*sini

    else:
        if ecc != 0:
            raise NotImplementedError
        # analytic solution for circular orbits (from radvel.kepler)

        per = np.exp(logP)
        tp = t0
        om = 0
        k = np.exp(logK)

        m = 2 * np.pi * (((t_observed - tp) / per) - np.floor((t_observed - tp) / per))

        _rv = k * sini * np.cos(m + om)

    coef = polyfit(t_observed, _rv, 1)

    slope = coef[0]

    isdetectable = np.abs(slope) > gammadot_limit.to(u.m/u.s/u.day).value

    return slope, isdetectable




def calc_semiamplitude_period_recovery(N_samples, gammadot_limit, delta_time,
                                       t_observed, do_serial=0):

    log_semiamplitude = np.random.uniform(
        np.log(1e1), np.log(1e7), size=N_samples
    )

    log_period = np.random.uniform(
        np.log(1e0), np.log(1e15), size=N_samples
    )

    cosi = np.random.uniform(0, 1, N_samples)
    i = np.arccos(cosi)
    sini = np.sin(i)

    phase = np.random.uniform(0, 1, N_samples)

    t_start = t_observed.min()

    t0s = t_start + phase * np.exp(log_period)

    outpath = os.path.join(
        rdir, f'semiamplitude_period_recovery_{N_samples}.csv'
    )

    if not os.path.exists(outpath):

        if do_serial:

            slopes, detectables = [], []

            ix = 0
            for logK, logP, t0, si in zip(
                log_semiamplitude, log_period, t0s, sini
            ):

                if ix % 10 == 0:
                    print(f'{ix}/{N_samples}')

                task = (logK, logP, t0, si, t_observed, gammadot_limit)

                slope, isdetectable = rv_injection_worker(task)

                slopes.append(slope)
                detectables.append(isdetectable)

                ix += 1

        else:

            print(f'{datetime.now().isoformat()}: Beginning injection workers')

            tasks = [
                (logK, logP, t0, si, t_observed, gammadot_limit) for
                logK, logP, t0, si in
                zip(log_semiamplitude, log_period, t0s, sini)
            ]

            nworkers = mp.cpu_count()
            maxworkertasks = 1000
            pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

            result = pool.map(rv_injection_worker, tasks)

            pool.close()
            pool.join()

            print(f'{datetime.now().isoformat()}: Finished injection workers')

            out = np.array(result, dtype=np.dtype('float,bool'))
            out.dtype.names = ['slopes','detectables']
            slopes = out['slopes']
            detectables = out['detectables']

        df = pd.DataFrame({
            'logK': log_semiamplitude,
            'logP': log_period,
            'phase': phase,
            'slope': slopes,
            'detectable': detectables
        })
        df.to_csv(outpath, index=False)
        print(f'saved {outpath}')

    else:
        df = pd.read_csv(outpath)

    #
    # make plots; convert to desired mass vs semimajor axis contours
    #

    from aesthetic.plot import set_style, savefig
    set_style()
    f,ax = plt.subplots()
    ax.scatter(df.logP, df.logK, c=df.detectable.astype(int), s=0.2)
    ax.set_xlabel('logP'); ax.set_ylabel('logK')
    figpath = os.path.join(rdir, 'logP_vs_logK_detectable_check.png')
    savefig(f, figpath, writepdf=False, dpi=300)

    # NOTE: assuming log[k (m/s)]
    # NOTE: this is the Mstar >> Mcomp approximation. If this doesn't hold, it's an
    # implicit equation (the "binary mass function"). So in the large semimajor
    # axis regime, your resulting converted contours are slightly wrong.
    e = 0
    sini = 1
    Mstar = MSTAR * u.Msun

    msini_msun = ((
        (np.exp(np.array(df.logK)) / (28.4329)) * (1-e**2)**(1/2) *
        (Mstar.to(u.Msun).value)**(2/3) *
        ((np.exp(np.array(df.logP))*u.day).to(u.yr).value)**(1/3)
    )*u.Mjup).to(u.Msun).value

    log10msini = np.log10(msini_msun)

    per1 = np.exp(np.array(df.logP))*u.day
    sma1 = ((
        per1**2 * const.G*(Mstar + msini_msun*u.Msun) / (4*np.pi**2)
    )**(1/3)).to(u.au).value

    log10sma1 = np.log10(sma1)

    plt.close('all')
    f,ax = plt.subplots()
    ax.scatter(log10sma1, log10msini, c=df.detectable.astype(int), s=0.2)
    ax.set_xlabel('log$_{10}$(semi-major axis [AU])')
    ax.set_ylabel('log$_{10}$(M$\sin i$ [M$_\odot$])')
    ax.set_xlim([-2, 5]); ax.set_ylim([np.log10(0.001), np.log10(2)])
    figpath = os.path.join(rdir, 'log10sma_vs_log10msini_detectable_check.png')
    savefig(f, figpath, writepdf=False, dpi=300)


def get_abs_m(overwrite=0):
    # overwrite: whether to overwrite the pickle (~20s sampling)

    datestr = '20200624'
    cleanrvpath = os.path.join(DATADIR, 'spectra', 'RVs_{}_clean.csv'.format(datestr))
    df = pd.read_csv(cleanrvpath)

    # NOTE: the FEROS data contain all the information about the long-term
    # trend.
    sel = (df.tel == "FEROS")
    df = df[sel]
    df = df.sort_values(by='time')

    delta_t = (df.time.max() - df.time.min())
    t0 = df.time.min() + delta_t/2
    df['x'] = df['time'] - t0 # np.nanmean(df['time'])
    df['y'] = df['mnvel'] - np.nanmean(df['mnvel'])
    df['y_err'] = df['errvel']
    t_observed = np.array(df.time)

    force_err = 100
    print('WRN: inflating error bars to account for rot jitter')
    print(f'{df.y_err.median()} to {force_err}')
    df.y_err = force_err

    pklpath = os.path.join(rdir, 'rvlim_method2.pkl')
    if os.path.exists(pklpath) and overwrite:
        os.remove(pklpath)

    # y = mx + c
    if not os.path.exists(pklpath):
        with pm.Model() as model:
            # Define priors
            c = pm.Uniform('c', lower=-100, upper=100)
            m = pm.Uniform('m', lower=-100, upper=100)

            abs_m = pm.Deterministic(
                "abs_m", pm.math.abs_(m)
            )

            # Define likelihood
            # Here Y ~ N(Xβ, σ^2), for β the coefficients of the model. Note though
            # the error bars are not _observed_ in this case; they are part of the
            # model!
            likelihood = pm.Normal('y', mu=m*nparr(df.x) + c,
                                   sigma=nparr(df.y_err), observed=nparr(df.y))

            # Inference!  draw 1000 posterior samples using NUTS sampling
            n_samples = 6000
            trace = pm.sample(n_samples, cores=16)

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace}, buff)

    else:
        d = pickle.load(open(pklpath, 'rb'))
        model, trace = d['model'], d['trace']

    # corner
    trace_df = trace_to_dataframe(trace)
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True)
    fig.savefig(os.path.join(rdir, 'corner.png'))

    # data + model
    plt.close('all')
    plt.figure(figsize=(7, 7))
    plt.scatter(df.x, df.y, label='data', zorder=2, color='k')
    plt.errorbar(df.x, df.y, yerr=df.y_err, ecolor='k', elinewidth=1,
                 capsize=2, zorder=2, ls='none')

    N_samples = 100
    lm = lambda x, sample: sample['m'] * x + sample['c']
    for rand_loc in np.random.randint(0, len(trace), N_samples):
        rand_sample = trace[rand_loc]
        plt.plot(nparr(df.x), lm(nparr(df.x), rand_sample), zorder=1,
                 alpha=0.5, color='C0')

    plt.legend(loc=0)
    plt.xlabel('time [d]')
    plt.ylabel('rv [m/s]')
    plt.savefig(os.path.join(rdir, 'datamodel.png'))
    plt.close('all')

    printparams = ['c', 'm', 'abs_m']
    print(42*'-')
    for p in printparams:
        med = np.percentile(trace[p], 50)
        up = np.percentile(trace[p], 84)
        low = np.percentile(trace[p], 36)
        threesig = np.percentile(trace[p], 99.7)
        print(f'{p} : {med:.3f} +{up-med:.3f} -{med-low:.3f}')
        print(f'{p} 99.7: {threesig:.3f}')
    print(42*'-')

    absm_threesig = threesig*u.m/u.s/u.day
    delta_time = df.time.max() - df.time.min()

    return absm_threesig, delta_time, t_observed


def main(overwrite=0):

    # steps 1 and 2. [m] = m/s/day
    gammadot_limit, delta_time, t_observed = get_abs_m(overwrite=overwrite)
    print(f'BASELINE is {delta_time:.2f} days')

    # 3. for a given value of P, what value of K would yield a first derivative
    # greater than the slope limit most of the time?
    N_samples = int(2e5)

    calc_semiamplitude_period_recovery(N_samples, gammadot_limit, delta_time,
                                       t_observed)

    # actual sensitivity is then hand-interpolated from
    # results/rvlimits_method2/log10sma_vs_log10msini_detectable_check.png

    handdrawn_sensitivity_df = pd.read_csv(
        os.path.join(rdir, 'rvoutersensitivty_method2.csv'), comment='#'
    )

    outdf = pd.DataFrame({
        'log10sma': handdrawn_sensitivity_df.log10sma,
        'log10mpsini': handdrawn_sensitivity_df.log10msini
    })
    outdf = outdf.sort_values(by='log10sma')

    outpath = '../../results/fpscenarios/rvoutersensitivity_method2_3sigma.csv'

    outdf.to_csv(outpath, index=False)
    print(f'made {outpath}')



if __name__ == "__main__":
    main()
