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
from scipy.stats import gaussian_kde

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
    get ground-based photometry from Phil Evans.
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


def get_astep_phot(datestr=None):
    """
    get ground-based photometry from ASTEP400

    datestrs = ['20200529', '20200614', '20200623']
    """

    if datestr is None:
        raise NotImplementedError

    else:
        lcglob = os.path.join(RESULTSDIR, 'groundphot', 'externalreduc',
                              'ASTEP_to_fit', f'TIC*{datestr}*.csv')
        lcpaths = glob(lcglob)
        assert len(lcpaths) == 1

    time, flux, flux_err = [], [], []

    for l in lcpaths:

        df = pd.read_csv(l)

        time_k = 'BJD'
        flux_k = 'FLUX'
        flux_err_k = 'ERRFLUX'

        time.append(nparr(df[time_k]))
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
    except KeyError:
        r = np.exp(paramd['log_r'])

    b = paramd['b']
    u0 = paramd['u[0]']
    u1 = paramd['u[1]']

    r_star = paramd['r_star']
    logg_star = paramd['logg_star']

    try:
        mean = paramd['mean']
    except KeyError:
        mean_key = [k for k in list(paramd.keys()) if 'mean' in k]
        assert len(mean_key) == 1
        mean_key = mean_key[0]
        mean = paramd[mean_key]

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


def get_model_transit_quad(paramd, time_eval, _tmid, t_exp=2/(60*24),
                           includemean=1):
    """
    Same as get_model_transit, but for a transit + quadratic trend. The midtime
    of the trend must be the same as used in timmy.modelfitter for the a1 and
    a2 coefficients to be correctly defined.
    """
    import exoplanet as xo

    period = paramd['period']
    t0 = paramd['t0']
    try:
        r = paramd['r']
    except KeyError:
        r = np.exp(paramd['log_r'])

    b = paramd['b']
    u0 = paramd['u[0]']
    u1 = paramd['u[1]']

    r_star = paramd['r_star']
    logg_star = paramd['logg_star']

    try:
        mean = paramd['mean']
    except KeyError:
        mean_key = [k for k in list(paramd.keys()) if 'mean' in k]
        assert len(mean_key) == 1
        mean_key = mean_key[0]
        mean = paramd[mean_key]

    a1 = paramd[ [k for k in list(paramd.keys()) if '_a1' in k][0] ]
    a2 = paramd[ [k for k in list(paramd.keys()) if '_a2' in k][0] ]

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

    mu_model = (
        mean +
        a1*(time_eval-_tmid) +
        a2*(time_eval-_tmid)**2 +
        mu_transit.eval()
    )

    if includemean:
        mu_trend = (
            mean +
            a1*(time_eval-_tmid) +
            a2*(time_eval-_tmid)**2
        )
    else:
        mu_trend = (
            a1*(time_eval-_tmid) +
            a2*(time_eval-_tmid)**2
        )

    return mu_model, mu_trend


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


def _get_fitted_data_dict_alltransit(m, summdf):

    d = OrderedDict()

    for name in m.data.keys():

        d[name] = {}
        d[name]['x_obs'] = m.data[name][0]
        d[name]['y_obs'] = m.data[name][1]
        d[name]['y_err'] = m.data[name][2]

        params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]', 'r_star',
                  'logg_star', f'{name}_mean']

        paramd = {k : summdf.loc[k, 'median'] for k in params}
        y_mod_median = get_model_transit(paramd, d[name]['x_obs'])

        d[name]['y_mod'] = y_mod_median
        d[name]['y_resid'] = d[name]['y_obs'] - y_mod_median
        d[name]['params'] = params

    return d


def _estimate_mode(samples, N=1000):
    """
    Estimates the "mode" (really, maximum) of a unimodal probability
    distribution given samples from that distribution. Do it by approximating
    the distribution using a gaussian KDE, with an auto-tuned bandwidth that
    uses Scott's rule of thumb.
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>

    args:

        samples: 1d numpy array of sampled values

        N: number of points at which to evalute the KDE. higher improves
        precision of the estimate.

    returns:

        Peak of the distribution. (Assuming it is unimodal, which should be
        checked.)
    """

    kde = gaussian_kde(samples, bw_method='scott')

    x = np.linspace(min(samples), max(samples), N)

    probs = kde.evaluate(x)

    peak = x[np.argmax(probs)]

    return peak



def _get_fitted_data_dict_allindivtransit(m, summdf, bestfitmeans='median'):
    """
    args:
        bestfitmeans: "map", "median", "mean, "mode"; depending on which you
        think will produce the better fitting model.
    """

    d = OrderedDict()

    for name in m.data.keys():

        d[name] = {}
        d[name]['x_obs'] = m.data[name][0]
        # d[name]['y_obs'] = m.data[name][1]
        d[name]['y_err'] = m.data[name][2]

        params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]', 'r_star',
                  'logg_star', f'{name}_mean', f'{name}_a1', f'{name}_a2']

        _tmid = np.nanmedian(m.data[name][0])
        t_exp = np.nanmedian(np.diff(m.data[name][0]))

        if bestfitmeans == 'mode':
            paramd = {}
            for k in params:
                print(name, k)
                paramd[k] = _estimate_mode(m.trace[k])
        elif bestfitmeans == 'median':
            paramd = {k : summdf.loc[k, 'median'] for k in params}
        elif bestfitmeans == 'mean':
            paramd = {k : summdf.loc[k, 'mean'] for k in params}
        elif bestfitmeans == 'map':
            paramd = {k : m.map_estimate[k] for k in params}
        else:
            raise NotImplementedError

        y_mod_median, y_mod_median_trend = (
            get_model_transit_quad(paramd, d[name]['x_obs'], _tmid,
                                   t_exp=t_exp, includemean=1)
        )

        # this is used for phase-folded data, with the local trend removed.
        d[name]['y_mod'] = y_mod_median - y_mod_median_trend

        # NOTE: for this case, the "residual" of the observation minus the
        # quadratic trend is actually the "observation". this is b/c the
        # observation includes the rotation signal.
        d[name]['y_obs'] = m.data[name][1] - y_mod_median_trend

        d[name]['params'] = params

    # merge all the tess transits
    n_tess = len([k for k in d.keys() if 'tess' in k])
    d['tess'] = {}
    d['all'] = {}
    _p = ['x_obs', 'y_obs', 'y_err', 'y_mod']
    for p in _p:
        d['tess'][p] = np.hstack([d[f'tess_{ix}'][p] for ix in range(n_tess)])
    for p in _p:
        d['all'][p] = np.hstack([d[f'{k}'][p] for k in d.keys() if '_' in k])

    return d


def _subset_cut(x_obs, y_flat, y_err, n=12, onlyodd=False, onlyeven=False):
    """
    n: [ t0 - n*tdur, t + n*tdur ]
    """

    t0 = 1574.2727299
    per = 8.3248321
    tdur = 2.0/24 # roughly
    epochs = np.arange(-100,100,1)
    mid_times = t0 + per*epochs

    sel = np.zeros_like(x_obs).astype(bool)
    for tra_ind, mid_time in zip(epochs, mid_times):
        if onlyeven:
            if tra_ind % 2 != 0:
                continue
        if onlyodd:
            if tra_ind % 2 != 1:
                continue

        start_time = mid_time - n*tdur
        end_time = mid_time + n*tdur
        s = (x_obs > start_time) & (x_obs < end_time)
        sel |= s

    print(42*'#')
    print(f'Before subset cut: {len(x_obs)} observations.')
    print(f'After subset cut: {len(x_obs[sel])} observations.')
    print(42*'#')

    x_obs = x_obs[sel]
    y_flat = y_flat[sel]
    y_err = y_err[sel]

    return x_obs, y_flat, y_err



