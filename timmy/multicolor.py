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

DELTA_OBS_TESS = 4374e-6

# DELTA_LIM_RC = 0.003208  # 1-sigma lower limit on Rc-band depth
# DELTA_LIM_B = 0.002765  # 1-sigma lower limit on B-band depth
DELTA_LIM_RC = 0.002817  # 2-sigma lower limit on Rc-band depth
DELTA_LIM_B = 0.001769  # 2-sigma lower limit on B-band depth

def run_bulk_depth_color_grids(bandpass_to_use=None,
                               delta_obs_TESS=DELTA_OBS_TESS,
                               delta_lim_Rc=DELTA_LIM_RC,
                               delta_lim_B=DELTA_LIM_B, overwrite=0):
    """
    bandpass_to_use: 'Rc' or 'B'

    delta_obs_Rc: N-sigma lower limit
        (note: the difference between say a limit of 30ppt and 35ppt is a lot!)

    delta_obs_B: N-sigma lower limit
    """

    assert bandpass_to_use in ['Rc','B']
    bp = bandpass_to_use

    outpath = f'../results/fpscenarios/multicolor_{bp:s}.csv'

    if not os.path.exists(outpath) and not overwrite:

        np.random.seed(42)

        m2s = np.arange(0.20, 1.10, 0.01)
        frac_viable = []

        for m2 in m2s:
            print(f'{m2:.2f}')

            N_sample = 20
            m3s = np.linspace(0.01, m2, N_sample)

            ddicts = {}
            for ix, m3 in enumerate(m3s):
                ddict = get_delta_obs_given_mstars(m2, m3, verbose=0)
                ddicts[ix] = ddict

            df = pd.DataFrame(ddicts).T

            df['m2'] = m2
            df['m3'] = m3s

            # if the total eclipse depth is not greater than the observed
            # depth, that means no geometric fudging can lead to this scenario
            # being plausible.
            df['isviable_TESS'] = df['TESS'] > delta_obs_TESS

            # geometric scaling factor
            scalefactor = delta_obs_TESS / df['TESS']
            df['TESS_scaled'] = df['TESS'] * scalefactor

            if bp == 'Rc':
                # for the HEB scenario to be plausible given  the depth observed in
                # Rc band
                df['Cousins_R_scaled'] = df['Cousins_R'] * scalefactor
                df['isviable_Cousins_R'] = df['Cousins_R_scaled'] > delta_lim_Rc
                N_viable = len(df[df['isviable_TESS'] & df['isviable_Cousins_R']])
            elif bp == 'B':
                df['Johnson_B_scaled'] = df['Johnson_B'] * scalefactor
                df['isviable_Johnson_B'] = df['Johnson_B_scaled'] > delta_lim_B
                N_viable = len(df[df['isviable_TESS'] & df['isviable_Johnson_B']])
            else:
                raise NotImplementedError

            N_tot = len(df)
            frac_viable.append(N_viable / N_tot)

        frac_viable = nparr(frac_viable)

        outdf = pd.DataFrame({
            'm2':m2s,
            'frac_viable':frac_viable
        })

        outdf.to_csv(outpath, index=False)

    else:
        print(f'found {outpath}')


def _find_nearest(df, param, mass):

    mass_ind = np.argmin(np.abs(nparr(df.mass) - mass))

    return df.loc[mass_ind, param]


def get_delta_obs_given_mstars(m2, m3, m1=1.1, make_plot=0, verbose=1):
    """
    You are given a hierarchical eclipsing binary system.
    Star 1 (TOI 837) is the main source of light.
    Stars 2 and 3 are eclipsing.

    Work out the observed eclipse depth of Star 3 in front of Star 2 in each of
    a number of bandpasses, assuming maximally large eclipses, and blackbodies.
    Ah, and also assuming MIST isochrones.
    """

    # Given the stellar masses, get their effective temperatures and
    # luminosities.
    #
    # At ~50 Myr, M dwarf companion PMS contraction will matter for its
    # parameters. Use
    # data.companion_isochrones.MIST_plus_Baraffe_merged_3.5e+07.csv,
    # which was created during the dmag to companion mass conversion process.
    #
    mstars = nparr([m1, m2, m3])

    icdir = os.path.join(DATADIR, 'companion_isochrones')
    df_ic = pd.read_csv(
        os.path.join(icdir, 'MIST_plus_Baraffe_merged_3.5e+07.csv')
    )

    teff1 = TEFF
    teff2 = _find_nearest(df_ic, 'teff', m2)
    teff3 = _find_nearest(df_ic, 'teff', m3)

    lum1 = (4*np.pi*(RSTAR*u.Rsun)**2 * const.sigma_sb*(TEFF*u.K)**4).to(u.Lsun).value
    lum2 = _find_nearest(df_ic, 'lum', m2)
    lum3 = _find_nearest(df_ic, 'lum', m3)

    teffs = nparr([teff1, teff2, teff3])
    lums = nparr([lum1, lum2, lum3])

    #
    # initialization. other working bandpasses include Johnson_U, Johnson_V,
    # SDSS_g., and SDSS_z.
    #
    bpdir = os.path.join(DATADIR, 'bandpasses')
    bandpasses = [
        'Bessell_U', 'Bessell_B', 'Bessell_V', 'Cousins_R', 'Cousins_I',
        'TESS', 'Johnson_B'
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

        #
        # for each star, calculate erg/s in bandpass
        # NOTE: the quantity of interest is in fact counts/sec. 
        # (this is probably a small consideration, but could still be worth
        # checking)
        #

        rstars = []

        for ix, temperature, luminosity in zip(
            range(len(teffs)), teffs*u.K, lums*u.Lsun
        ):

            # flux (erg/s/cm^2) in bandpass
            _F_X = (
                4*np.pi*u.sr *
                trapz(B_lambda_dict[ix] * T_lambda_interp, wvlen)
            )

            F_X_dict[bp][ix] = _F_X

            # bolometric flux, according to the blackbody function
            _F_bol = (
                4*np.pi*u.sr * trapz(B_lambda_dict[ix] *
                                     np.ones_like(T_lambda_interp), wvlen)
            )

            # stefan-boltzman law to get rstar from the isochronal temp/lum.
            rstar = np.sqrt(
                luminosity /
                (4*np.pi * const.sigma_sb * temperature**4)
            )
            rstars.append(rstar.to(u.Rsun).value)

            # erg/s in bandpass
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

        # out of eclipse
        L_ooe = L_X_dict[k][0] + L_X_dict[k][1] + L_X_dict[k][2]

        # in eclipse. assume maximal depth
        f = (rstars[2]/rstars[1])**2
        L_ie = L_X_dict[k][0] + L_X_dict[k][2] + (1-f)*L_X_dict[k][1]

        # assume the tertiary is eclipsing the secondary.
        delta_obs_dict[k] = (
            (L_ooe - L_ie)/L_ooe
        ).value

    if verbose:
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
