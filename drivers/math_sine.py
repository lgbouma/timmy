import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units as u

def simple(period=100*u.day, gammadot_limit=0.82*u.m/u.s/u.day, frac=0.997):
    """
    Given a period and limiting dv/dt,
    return the minimum K such that

        |K ω sin(ωt)| < gammadot_limit

    at least frac % of the time. (Frac is expressed where 99.7% = 0.997)

    NOTE: this code convinced me that when frac is sufficiently close to 1,
    given the shape of the arcsine distribution, a fine enough approximation
    (in fact, a lower bound), is to just give that minimum K as:

        gammadot_limit / ω
    """

    N_draws = 10000

    t = np.linspace(0, period.to(u.day).value, N_draws, endpoint=True)
    ω = 2*np.pi/(period.to(u.day).value)

    K_init = (gammadot_limit.to(u.m/u.s/u.day).value) / ω

    K = K_init

    y = K*ω*np.abs(np.sin(ω*t))

    plt.close('all')
    _bins = np.linspace(0, K*ω, 1000)
    n, bins, patches = plt.hist(y, bins=_bins, density=True, cumulative=True)
    plt.vlines(gammadot_limit.value, 0, 1)
    plt.xlabel('Kω|sin(ωt)| [m/s/day]')
    plt.ylabel('cumulative frac')
    plt.savefig('../results/rvlimits_method2/math_sine.png')
    plt.close('all')

    binmids = bins[:-1] + np.diff(bins)/2

    fn = interp1d(n, binmids)

    foo = fn(frac)
    print(foo)
    print(K)


if __name__ == '__main__':
    simple()
