import numpy as np, pandas as pd
import collections, pickle, os
from glob import glob
from collections import OrderedDict
from astropy.io import fits

from astrobase.lcmath import time_bin_magseries_with_errs
from cdips.lcproc.mask_orbit_edges import mask_orbit_start_and_end
from cdips.plotting.vetting_pdf import _given_mag_get_flux

from timmy.paths import DATADIR, RESULTSDIR

from numpy import array as nparr

def detrend_tessphot(x_obs, y_obs, y_err):

    from wotan import flatten

    flat_flux, trend_flux = flatten(x_obs, y_obs, method='hspline',
                                    window_length=0.3,
                                    break_tolerance=0.4, return_trend=True)

    # flat_flux, trend_flux = flatten(time, flux, method='pspline',
    #                                 break_tolerance=0.4, return_trend=True)
    # flat_flux, trend_flux = flatten(time, flux, method='biweight',
    #                                 window_length=0.3, edge_cutoff=0.5,
    #                                 break_tolerance=0.4, return_trend=True,
    #                                 cval=2.0)

    return flat_flux, trend_flux



def get_tessphot(provenance, yval):
    """
    provenance: 'spoc' or 'cdips'

    yval:
        spoc: 'SAP_FLUX', 'PDCSAP_FLUX'
        cdips: 'PCA1', 'IRM1', etc.
    """

    if provenance == 'spoc':
        lcpaths = glob(os.path.join(
            DATADIR, 'MAST_2020-05-04T1852/TESS/*/*-s_lc.fits'))
        assert len(lcpaths) == 2
    elif provenance == 'cdips':
        lcpaths = glob(os.path.join(
            DATADIR, 'MAST_2020-05-04T1852/HLSP/*/*cdips*llc.fits'))
        assert len(lcpaths) == 2
    else:
        raise NotImplementedError

    time, flux, flux_err, qual = [], [], [], []

    for l in lcpaths:

        hdul = fits.open(l)
        d = hdul[1].data

        if provenance == 'spoc':
            time.append(d['TIME'])
            _f, _f_err = d[yval], d[yval+'_ERR']
            flux.append(_f/np.nanmedian(_f))
            flux_err.append(_f_err/np.nanmedian(_f))
            qual.append(d['QUALITY'])

        elif provenance == 'cdips':

            time.append(d['TMID_BJD'] - 2457000)
            _f, _f_err = _given_mag_get_flux(d[yval], err_mag=d['IRE'+yval[-1]])
            flux.append(_f)
            flux_err.append(_f_err)

        hdul.close()

    time = np.concatenate(time).ravel()
    flux = np.concatenate(flux).ravel()
    flux_err = np.concatenate(flux_err).ravel()
    if len(qual)>0:
        qual = np.concatenate(qual).ravel()

    return time, flux, flux_err, qual



def get_clean_tessphot(provenance, yval, binsize=None, maskflares=0):
    """
    Get data. Mask quality cut.

    Optionally bin, to speed fitting (linear in time, but at the end of the
    day, you want 2 minute).
    """

    time, flux, flux_err, qual = get_tessphot(provenance, yval)

    N_i = len(time) # initial

    if provenance == 'spoc':

        # [   0,    1,    8,   16,   32,  128,  160,  168,  176,  180,  181,
        #   512, 2048, 2080, 2176, 2216, 2560]
        binary_repr_vec = np.vectorize(np.binary_repr)
        qual_binary = binary_repr_vec(qual, width=12)

        # See Table 28 of EXP-TESS-ARC-ICD-TM-0014
        # don't want:
        # bit 3 coarse point
        # bit 4 Earth point
        # bit 6 reaction wheel desaturation
        # bit 8 maunal exclude
        # bit 11 cosmic ray detected on collateral pixel row or column
        # bit 12 straylight from earth or moon in camera fov

        # badbits = [3,4,6,8,11]
        badbits = [3,4,6,8,11,12]

        sel = np.isfinite(qual)
        for bb in badbits:
            # note zero vs one-based count here to convert bitwise flags to
            # python flags
            sel &= ~(np.array([q[bb-1] for q in qual_binary]).astype(bool))

        time, flux, flux_err = time[sel], flux[sel], flux_err[sel]

        inds = np.argsort(time)
        time, flux, flux_err = time[inds], flux[inds], flux_err[inds]

    N_ii = len(time) # after quality cut

    # finite times, fluxes, flux errors.
    sel = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[sel], flux[sel], flux_err[sel]

    N_iii = len(time)

    # time, flux, sel = mask_orbit_start_and_end(time, flux, orbitgap=0.5,
    #                                            expected_norbits=2,
    #                                            orbitpadding=6/(24),
    #                                            raise_expectation_error=True,
    #                                            return_inds=True)
    # flux_err = flux_err[sel]

    # N_iii = len(time) # after orbit edge masking

    if maskflares:

        t_offset = np.nanmin(time)

        FLARETIMES = [
            (4.60+t_offset, 4.63+t_offset),
            (37.533+t_offset, 37.62+t_offset)
        ]

        flaresel = np.zeros_like(time).astype(bool)
        for ft in FLARETIMES:
            flaresel |= ( (time > min(ft)) & (time < max(ft)) )

        time, flux, flux_err = (
            time[~flaresel], flux[~flaresel], flux_err[~flaresel]
        )

        N_iv = len(time)


    x_obs = time
    y_obs = (flux / np.nanmedian(flux))
    y_err = flux_err / np.nanmedian(flux)

    print(42*'-')
    print('N initial: {}'.
          format(N_i))
    print('N after quality cut: {}'.
          format(N_ii))
    print('N after quality cut + finite masking: {}'.
          format(N_iii))
    if maskflares:
        print('N after quality cut + finite masking + flare masking: {}'.
              format(N_iv))
    print(42*'-')

    if isinstance(binsize, int):
        bd = time_bin_magseries_with_errs(
            x_obs, y_obs, y_err, binsize=binsize, minbinelems=5
        )
        x_obs = bd['binnedtimes']
        y_obs = bd['binnedmags']

        # assume errors scale as sqrt(N)
        original_cadence = 120
        y_err = bd['binnederrs'] / (binsize/original_cadence)**(1/2)

    assert len(x_obs) == len(y_obs) == len(y_err)

    return (
        x_obs.astype(np.float64),
        y_obs.astype(np.float64),
        y_err.astype(np.float64)
    )


def get_elsauce_phot(datestr=None):
    """
    concatenate ground-based photometry from Phil Evans (2020.04.01,
    2020.04.26, and 2020.05.21).

    2020-04-01: R_c
    2020-04-26: R_c
    2020-05-21: I_c
    2020-06-14: B_j
    """

    if datestr is None:
        lcglob = os.path.join(RESULTSDIR, 'groundphot', 'externalreduc',
                              'bestkaren', 'to_fit', '*.dat')
        lcpaths = glob(lcglob)
        assert len(lcpaths) == 4

    else:
        lcglob = os.path.join(RESULTSDIR, 'groundphot', 'externalreduc',
                              'bestkaren', 'to_fit', f'TIC*{datestr}*.dat')
        lcpaths = glob(lcglob)
        assert len(lcpaths) == 1

    time, flux, flux_err = [], [], []

    for l in lcpaths:

        df = pd.read_csv(l, delim_whitespace=True)

        time.append(nparr(df['BJD_TDB']))

        if 'rel_flux_T1_dfn' in df:
            flux_k = 'rel_flux_T1_dfn'
            flux_err_k = 'rel_flux_err_T1_dfn'
        else:
            flux_k = 'rel_flux_T1_n'
            flux_err_k = 'rel_flux_err_T1_n'

        flux.append(nparr(df[flux_k]))
        flux_err.append(nparr(df[flux_err_k]))

    time = np.concatenate(time).ravel()
    flux = np.concatenate(flux).ravel()
    flux_err = np.concatenate(flux_err).ravel()

    return time, flux, flux_err




def get_clean_rv_data(datestr='20200525'):
    # get zero-subtracted RV CSV in m/s units, time-sorted.

    rvpath = os.path.join(DATADIR, 'spectra', 'RVs_{}.csv'.format(datestr))
    df = pd.read_csv(rvpath)

    time = nparr(df['time'])
    mnvel = nparr(df['mnvel'])
    errvel = nparr(df['errvel'])
    telvec = nparr(df['tel'])
    source = nparr(df['Source'])

    # first, zero-subtract each instrument median. then, set units to be
    # m/s, not km/s.
    umedians = {}
    for uinstr in np.unique(telvec):
        umedians[uinstr] = np.nanmedian(mnvel[telvec == uinstr])
        mnvel[telvec == uinstr] -= umedians[uinstr]

    mnvel *= 1e3
    errvel *= 1e3

    # time-sort
    inds = np.argsort(time)

    time = np.ascontiguousarray(time[inds], dtype=float)
    mnvel = np.ascontiguousarray(mnvel[inds], dtype=float)
    errvel = np.ascontiguousarray(errvel[inds], dtype=float)
    telvec = np.ascontiguousarray(telvec[inds], dtype=str)
    source = np.ascontiguousarray(source[inds], dtype=str)

    cdf = pd.DataFrame({
        'time': time,
        'tel': telvec,
        'Name': nparr(df['Name']),
        'mnvel': mnvel,
        'errvel': errvel,
        'Source': source
    })

    cleanrvpath = os.path.join(DATADIR, 'spectra', 'RVs_{}_clean.csv'.format(datestr))
    if not os.path.exists(cleanrvpath):
        cdf.to_csv(cleanrvpath, index=False)

    return cdf



def get_model_transit(paramd, time_eval, t_exp=2/(60*24)):
    """
    you know the paramters, and just want to evaluate the median lightcurve.
    """
    import exoplanet as xo

    period = paramd['period']
    t0 = paramd['t0']
    try:
        r = paramd['r']
    except:
        r = np.exp(paramd['log_r'])
    b = paramd['b']
    u0 = paramd['u[0]']
    u1 = paramd['u[1]']
    mean = paramd['mean']
    r_star = paramd['r_star']
    logg_star = paramd['logg_star']

    # factor * 10**logg / r_star = rho
    factor = 5.141596357654149e-05

    rho_star = factor*10**logg_star / r_star

    orbit = xo.orbits.KeplerianOrbit(
        period=period, t0=t0, b=b, rho_star=rho_star
    )

    u = [u0, u1]

    mu_transit = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=time_eval, texp=t_exp
    ).T.flatten()

    return mu_transit.eval() + mean


def _write_vespa(time, flux, flux_err):

    from astrobase.lcmath import phase_magseries_with_errs

    t0 = 1574.2727299
    period = 8.3248321

    orb_d = phase_magseries_with_errs(
        time, flux, flux_err, period, t0, wrap=True, sort=True
    )

    t = orb_d['phase']*period*24
    f = orb_d['mags']
    e = orb_d['errs']

    # take points +/- 2 hours from transit
    s = (t > -2) & (t < 2)

    t,f,e = t[s],f[s],e[s]

    outdf = pd.DataFrame({'t':t, 'f':f, 'e':e})

    outdf.to_csv('vespa_drivers/dtr_837_lc.csv', index=False, header=False)

    print('wrote vespa lc')


def _get_fitted_data_dict(m, summdf):

    d = {
        'x_obs': m.x_obs,
        'y_obs': m.y_obs,
        'y_orb': m.y_obs, # NOTE: "detrended" beforehand
        'y_resid': None, # for now.
        'y_mod': None, # for now [could be MAP, if MAP were good]
        'y_err': m.y_err
    }

    params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]', 'mean', 'r_star',
              'logg_star']
    paramd = {k:summdf.loc[k, 'median'] for k in params}
    y_mod_median = get_model_transit(paramd, d['x_obs'])
    d['y_mod'] = y_mod_median
    d['y_resid'] = m.y_obs-y_mod_median

    return d, params, paramd
