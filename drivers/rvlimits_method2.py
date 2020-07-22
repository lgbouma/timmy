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
from astropy import units as u, constants as c

from timmy.paths import DATADIR, RESULTSDIR
rdir = os.path.join(RESULTSDIR, 'rvlimits_method2')

def get_K_lim(period, gammadot_limit, frac=0.997):
    """
    Given a period and limiting dv/dt,
    return the minimum K such that

        |K ω sin(ωt)| - gammadot_limit > 0

    at least frac % of the time.
    """

    N_draws = 10000
    N_req = int(frac*N_draws)

    t = np.linspace(0, period, N_draws)
    ω = 2*np.pi/period

    assert 0


    # rv_vals = 

    return K

def main(overwrite=0):

    # steps 1 and 2. [m] = m/s/day
    absm_threesig = get_abs_m(overwrite=overwrite)

    assert 0

    # step 3, get samples.
    # 3. for a given value of P, what value of K would yield a first derivative
    # greater than the slope limit more than 99.7% of the time.
    N_samples = 1000
    log_period = np.random.uniform(
        low=np.log(0.1), high=np.log(1e15), size=N_samples
    )


    e = 0
    sini = 1
    Mstar = 1.1 * u.Msun
    # NOTE: assuming log[k (m/s)]
    # NOTE: this is the Mstar >> Mcomp approximation. If this doesn't hold, it's an
    # implicit equation (the "binary mass function"). So in the large semimajor
    # axis regime, your resulting converted contours are wrong.
    msini_msun = ((
        (np.exp(logk1) / (28.4329)) * (1-e**2)**(1/2) *
        (Mstar.to(u.Msun).value)**(2/3) *
        ((np.exp(logper1)*u.day).to(u.yr).value)**(1/3)
    )*u.Mjup).to(u.Msun).value

    logmsini = np.log(msini_msun)

    per1 = np.exp(logper1)*u.day
    sma1 = ((
        per1**2 * const.G*(Mstar + msini_msun*u.Msun) / (4*np.pi**2)
    )**(1/3)).to(u.au).value

    logsma1 = np.log(sma1)




    radvel.prior.HardBounds('logper1', np.log(0.1), np.log(1e15)),





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

    return absm_threesig

if __name__ == "__main__":
    main()
