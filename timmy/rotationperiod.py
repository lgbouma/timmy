import numpy as np, matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from astropy.modeling import models, fitting
from aesthetic.plot import format_ax, savefig

def measure_rotation_period_and_unc(time, flux, period_min, period_max,
                                    period_fit_cut=0.5, nterms=1,
                                    samples_per_peak=50,
                                    plotpath=None):
    """
    You know a light curve contains rotational modulation. You want to get the
    period, and the uncertainty.  Get the period as the peak of the LS
    periodogram. Get the uncertainty by fitting a gaussian to that
    (oversampled) peak, and saying the std dev is the uncertainty.

    args:
        nterms (int): number of fourier terms in the periodogram

        period_fit_cut (float): the width in days of the periodogram to fit.
        essentially an upper limit on your period uncertainty.

        samples_per_peak (int): targetted samples per peak in the periodogram.
        astropy default is 5. to oversample, do more than 5.

    returns: period, period_unc
    """

    #
    # run initial lomb scargle
    #
    ls = LombScargle(time, flux, nterms=nterms)

    freq, power = ls.autopower(
        minimum_frequency=1/period_max, maximum_frequency=1/period_min,
        samples_per_peak=samples_per_peak
    )
    period = 1/freq

    ls_freq_0 = freq[np.argmax(power)]
    ls_period_0 = 1/ls_freq_0
    ls_power_0 = power[np.argmax(power)]

    #
    # now fit the gaussian, get the uncertainty as the oversampled width.
    #
    stddev_init = 0.1 # days
    g_init = models.Gaussian1D(amplitude=ls_power_0, mean=ls_period_0,
                               stddev=stddev_init)
    g_init.mean.fixed = True

    fit_g = fitting.LevMarLSQFitter()

    sel = (
        (period < ls_period_0 + period_fit_cut/2)
        &
        (period > ls_period_0 - period_fit_cut/2)
    )

    g = fit_g(g_init, period[sel], power[sel])

    if isinstance(plotpath, str):

        plt.close('all')

        fig = plt.figure(figsize=(6, 4))

        ax0 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=2)
        ax1 = plt.subplot2grid((2,2), (1,0), colspan=1)
        ax2 = plt.subplot2grid((2,2), (1,1), colspan=1)

        axs = [ax0, ax1, ax2]

        ax0.scatter(time, flux, c='k', zorder=3, s=0.75, rasterized=True,
                    linewidths=0)
        ax0.set_xlabel('time')
        ax0.set_ylabel('flux')

        ax1.plot(period, power)
        ax1.axvline(ls_period_0, alpha=0.4, lw=1, color='C0', ls='-')
        ax1.axvline(2*ls_period_0, alpha=0.4, lw=1, color='C0', ls='--')
        ax1.axvline(0.5*ls_period_0, alpha=0.4, lw=1, color='C0', ls='--')
        ax1.set_title(f'P = {ls_period_0:.3f} d')
        ax1.set_ylabel('power')
        ax1.set_xlabel('period')

        period_mod = np.linspace(min(period[sel]), max(period[sel]), 1000)
        ax2.plot(period[sel], power[sel])
        ax2.plot(period_mod, g(period_mod))
        ax2.set_title(f'P_unc = {g.stddev.value:.3f} d')
        ax2.set_xlabel('period')

        for a in axs:
            format(a)

        fig.tight_layout()
        savefig(fig, plotpath, writepdf=0, dpi=300)

    period = ls_period_0
    period_unc = g.stddev.value

    return period, period_unc
