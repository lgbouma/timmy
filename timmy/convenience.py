import numpy as np, pandas as pd
import collections, pickle, os
from glob import glob
from collections import OrderedDict
from astropy.io import fits

from astrobase.lcmath import time_bin_magseries_with_errs
from cdips.lcproc.mask_orbit_edges import mask_orbit_start_and_end
from cdips.plotting.vetting_pdf import _given_mag_get_flux

from timmy.paths import DATADIR

def detrend_data(x_obs, y_obs, y_err):

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



def get_data(provenance, yval):
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
            flux_err.append(_f/np.nanmedian(_f))
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



def get_clean_data(provenance, yval, binsize=None):
    """
    Get data. Mask quality cut.

    Optionally bin, to speed fitting (linear in time, but at the end of the
    day, you want 2 minute).
    """

    time, flux, flux_err, qual = get_data(provenance, yval)

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

    x_obs = time
    y_obs = (flux / np.nanmedian(flux))
    y_err = flux_err / np.nanmedian(flux)

    print(42*'-')
    print('N initial: {}'.format(N_i))
    print('N after quality cut: {}'.format(N_ii))
    print('N after quality cut + finite masking: {}'.format(N_iii))
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
