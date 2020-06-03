"""
We want to know for plausible HEB or BEB scenarios, what kind of eclipse depth
variations to expect over potentially observable bandpasses.
"""

import os
from glob import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from numpy import trapz
from scipy.interpolate import interp1d

from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy import constants as const

from timmy.priors import TEFF, RSTAR
from timmy.paths import DATADIR

DEBUG = 0

def main():

    np.random.seed(42)
    delta_obs_dict = get_delta_obs_given_mstars(0.205, 0.205)

    import IPython; IPython.embed()
    # TODO: move to the timmy package. add the currently working case as a test
    # case. test on different range of m1/m2.


def _find_nearest(df, param, mass):

    mass_ind = np.argmin(np.abs(nparr(df.mass) - mass))

    return df.loc[mass_ind, param]

def get_delta_obs_given_mstars(m2, m3, m1=1.1, make_plot=0):

    # At ~50 Myr, M dwarf companion PMS contraction will matter for its
    # parameters. Use those from
    # data.companion_isochrones.MIST_plus_Baraffe_merged_3.5e+07.csv
    mstars = nparr([m1, m2, m3])

    icdir = os.path.join(DATADIR, 'companion_isochrones')
    df_ic = pd.read_csv(
        os.path.join(icdir, 'MIST_plus_Baraffe_merged_3.5e+07.csv')
    )

    teff2 = _find_nearest(df_ic, 'teff', m2)
    teff3 = _find_nearest(df_ic, 'teff', m3)

    lum2 = _find_nearest(df_ic, 'lum', m2)
    lum3 = _find_nearest(df_ic, 'lum', m3)

    teffs = nparr([TEFF, teff2, teff3])
    lums = nparr(
        [(4*np.pi*(RSTAR*u.Rsun)**2 * const.sigma_sb*(TEFF*u.K)**4).to(u.Lsun).value,
          lum2,
          lum3]
    )

    #
    # initialization
    #
    bpdir = os.path.join(DATADIR, 'bandpasses')
    bandpasses = [
        'Bessell_U',
        #'Johnson_U',
        'Bessell_B',
        'Bessell_V',
        #'Johnson_V',
        #'SDSS_g.',
        'Cousins_R',
        'Cousins_I',
        #'SDSS_z.',
        'TESS'
    ]
    bppaths = [glob(os.path.join(bpdir, '*'+bp+'*csv'))[0] for bp in bandpasses]

    wvlen = np.logspace(1, 5, 2000)*u.nm

    #
    # do the calculation
    #

    B_lambda_dict = {}

    for ix, temperature, luminosity in zip(
        range(len(teffs)), teffs*u.K, lums*u.Lsun
    ):

        bb = BlackBody(temperature=temperature)

        B_nu_vals = bb(wvlen)

        B_lambda_vals = (
            B_nu_vals * (const.c / wvlen**2)
        ).to(u.erg/u.nm/u.s/u.sr/u.cm**2)

        B_lambda_dict[ix] = B_lambda_vals

    # now get the flux in each bandpass
    F_X_dict = {}
    F_dict = {}
    M_X_dict = {}
    M_dict = {}
    T_dict = {}
    L_X_dict = {}
    for bp, bppath in zip(bandpasses, bppaths):

        bpdf = pd.read_csv(bppath)
        if 'nm' in bpdf:
            pass
        else:
            bpdf['nm'] = bpdf['angstrom'] / 10

        bp_wvlen = nparr(bpdf['nm'])
        T_lambda = nparr(bpdf.Transmission)
        if np.nanmax(T_lambda) > 1.1:
            if np.nanmax(T_lambda) < 100:
                T_lambda /= 100 # unit convert
            else:
                raise NotImplementedError

        eps = 1e-6
        if not np.all(np.diff(bp_wvlen) > eps):
            raise NotImplementedError

        interp_fn = interp1d(bp_wvlen, T_lambda, bounds_error=False,
                             fill_value=0, kind='quadratic')

        T_lambda_interp = interp_fn(wvlen)
        T_dict[bp] = T_lambda_interp

        F_X_dict[bp] = {}
        M_X_dict[bp] = {}
        L_X_dict[bp] = {}
        # for each star, calculate flux in bandpass
        for ix, temperature, luminosity in zip(
            range(len(teffs)), teffs*u.K, lums*u.Lsun
        ):

            _F_X = 4*np.pi*u.sr * trapz(B_lambda_dict[ix] * T_lambda_interp, wvlen)
            F_X_dict[bp][ix] = _F_X

            _F_bol = 4*np.pi*u.sr * trapz(B_lambda_dict[ix] * np.ones_like(T_lambda_interp), wvlen)

            rstar = np.sqrt(
                luminosity /
                (4*np.pi * const.sigma_sb * temperature**4)
            )
            _L_X = _F_X * 4*np.pi * rstar**2

            L_X_dict[bp][ix] = _L_X.cgs

            if DEBUG:
                print(42*'-')
                print(f'{_F_X:.2e}, {_F_bol:.2e}, {L_X_dict[bp][ix]:.2e}')

            if ix not in F_dict.keys():
                F_dict[ix] = _F_bol

            # get bolometric magnitude of the star, in the bandpass, as a
            # sanity check
            M_bol_sun = 4.83
            M_bol_star = (
                -5/2 * np.log10(luminosity/(1*u.Lsun)) + M_bol_sun
            )
            M_X = M_bol_star - 5/2*np.log10( F_X_dict[bp][ix]/F_dict[ix] )

            if ix not in M_dict.keys():
                M_dict[ix] = M_bol_star.value
            M_X_dict[bp][ix] = M_X.value

    delta_obs_dict = {}
    for k in L_X_dict.keys():
        # NOTE: details are wrong here, if you care to distinguish
        # secondary and tertiary
        delta_obs_dict[k] = (
            L_X_dict[k][1] / (L_X_dict[k][0])
        ).value

    for k in delta_obs_dict.keys():
        print(f'{k}: {delta_obs_dict[k]:.2e}')

    if make_plot:
        #
        # plot the result
        #
        from astropy.visualization import quantity_support

        plt.close('all')
        linestyles = ['-','-','--']
        f, ax = plt.subplots(figsize=(4,3))
        with quantity_support():
            for ix in range(3):
                l = f'{teffs[ix]:d} K, {mstars[ix]:.3f} M$_\odot$'
                ax.plot(wvlen, B_lambda_dict[ix], ls=linestyles[ix],
                        label=l)
        ax.set_yscale('log')
        ax.set_ylim([1e-3,10*np.nanmax(np.array(list(B_lambda_dict.values())))])
        ax.set_xlabel('Wavelength [nm]')
        ax.legend(loc='best', fontsize='xx-small')

        ax.set_ylabel('$B_\lambda$ [erg nm$^{-1}$ s$^{-1}$ sr$^{-1}$ cm$^{-2}$ ]')
        tax = ax.twinx()
        tax.set_ylabel('Transmission [%]')
        for k in T_dict.keys():
            sel = T_dict[k] > 0
            tax.plot(wvlen[sel], 100*T_dict[k][sel], c='k', lw=0.5)
        tax.set_xscale('log')
        tax.set_ylim([0,105])

        ax.set_xlim([5e1, 1.1e4])

        f.tight_layout()
        f.savefig('../results/eclipse_depth_color_dependence/blackbody_transmission.png', dpi=300)

    return delta_obs_dict


if __name__ == "__main__":
    main()
