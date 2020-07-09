"""
Basic sanity checks for ephemeris.
"""
import numpy as np, pandas as pd
from numpy import array as nparr
from astrobase.timeutils import get_epochs_given_midtimes_and_period
from scipy.optimize import curve_fit

def linear_model(xdata, m, b):
    return m*xdata + b

def main():
    df = pd.read_csv('../../data/ephemeris/midtimes.csv')

    df = df.sort_values(by='tmid')

    tmid = nparr(df['tmid'])
    tmid_err = nparr(df['tmiderr'])

    P_orb_init = 8.32467 # SPOC, +/- 4e-4
    t0_orb_init = 2457000 + 1574.2738 # SPOC, +/- 1e-3

    epoch, _ = get_epochs_given_midtimes_and_period(
        tmid, P_orb_init, t0_fixed=t0_orb_init, verbose=True
    )

    popt, pcov = curve_fit(
        linear_model, epoch, tmid,
        p0=(P_orb_init, t0_orb_init),
        sigma=tmid_err
    )

    lsfit_period = popt[0]
    lsfit_period_err = pcov[0,0]**0.5
    lsfit_t0 = popt[1]
    lsfit_t0_err = pcov[1,1]**0.5

    print(f't0: {lsfit_t0:.6f} +/- {lsfit_t0_err:.6f}')
    print(f'period: {lsfit_period:.7f} +/- {lsfit_period_err:.7f}')

if __name__ == "__main__":
    main()
