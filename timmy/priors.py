import numpy as np
from collections import OrderedDict

# Stassun's SED analysis for Rstar. spectroscopic logg for Mstar.  Set the
# error bars xN, to let transit data should speak for itself, at least a bit
# more.

# RSTAR = 1.049
# RSTAR_STDEV = 0.019*3
# MSTAR = 1.21  # From logg and Rstar. Seems high.
# MSTAR_STDEV = 0.10*2
# LOGG = 4.48  # provenance: CHIRON spectroscopy/Zhou.
# LOGG_STDEV = 0.03*3
# TEFF = 5946 # CHIRON spectra

# FROM MIST ISOCHRONES ON IC 2602
RSTAR = 1.022
RSTAR_STDEV = 0.015
MSTAR = 1.118
MSTAR_STDEV = 0.011
LOGG = 4.467
LOGG_STDEV = 0.010
TEFF = 6047
TEFF_STDEV = 40

LI_EW = 154 # mA
P_ROT = 3.004 # days, from Lomb-Scargle fit.

ROT_AMP = 0.027 # i.e. 27 parts per thousands, peak-to-peak, by eye

VSINI = 17.483 # km/s, from average of 6 CHIRON spectra
VSINI_STDEV = 0.134 # km/s, ditto.

RV_PEAK_TO_PEAK_JITTER = VSINI*ROT_AMP
K_JITTER = RV_PEAK_TO_PEAK_JITTER * 0.5


def initialize_prior_d(modelcomponents, datasets=None):

    # P_orb = 8.32467 # SPOC, +/- 4e-4
    # t0_orb = 1574.2738 # SPOC, +/- 1e-3

    P_orb = 8.32489
    t0_orb = 1574.27380

    # P_orb = 8.3248972 # SG1, +/- 3.4e-4
    # t0_orb = 1574.2738304  # SG1, +/- 1.1e-3,  BTJD

    # P_orb = 8.328 # TESS + El Sauce fit
    # t0_orb = 1574.2646 # plausible transit

    rp_rs = 0.0865 # +/- 0.0303

    # Visual inspection
    P_rot = 3.3 # +/- 1 (by eye)
    t0_rot = None # idk, whatever phase. it's not even a model parameter
    amp = 1e3*1.5e-2 # primary mode amplitude [ppt]. peak to peak is a bit over 2%
    amp_mix = 0.5 # between 0 and 1
    log_Q0 = np.log(1e1) # Q of secondary oscillation. wide prior
    log_deltaQ = np.log(1e1) # primary mode gets higher quality

    prior_d = OrderedDict()

    for modelcomponent in modelcomponents:

        if 'alltransit' in modelcomponent:
            prior_d['period'] = P_orb
            prior_d['t0'] = t0_orb
            if 'quaddepthvar' not in modelcomponents:
                prior_d['log_r'] = np.log(rp_rs)
            else:
                # NOTE: this implementation is very 837-specific.
                prior_d['log_r_Tband'] = np.log(rp_rs)
                prior_d['log_r_Bband'] = np.log(rp_rs)
                prior_d['log_r_Rband'] = np.log(rp_rs)
            prior_d['b'] = 0.5  # initialize for broad prior

            prior_d['r_star'] = RSTAR
            prior_d['logg_star'] = LOGG

            # T-band Teff 5900K, logg 4.50 (Claret+18)
            prior_d['u[0]'] = 0.3249
            prior_d['u[1]'] = 0.235


            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                prior_d[f'{name}_mean'] = 1

        if 'quad' in modelcomponent:
            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                if name == 'tess':
                    pass
                else:
                    # model [per ground-transit] is :
                    # a0+ a1*(time-midtime) + a2*(time-midtime)^2.
                    # a0 is the mean, already above.
                    prior_d[f'{name}_a1'] = 0
                    prior_d[f'{name}_a2'] = 0

        if modelcomponent not in ['alltransit', 'onetransit',
                                  'allindivtransit']:
            if 'transit' in modelcomponent:
                prior_d['period'] = P_orb
                prior_d['t0'] = t0_orb
                # prior_d['r'] = rp_rs
                prior_d['log_r'] = np.log(rp_rs)
                prior_d['b'] = 0.5  # initialize for broad prior
                prior_d['u[0]'] = 0.3249
                prior_d['u[1]'] = 0.235
                prior_d['mean'] = 1
                prior_d['r_star'] = RSTAR
                prior_d['logg_star'] = LOGG

        if modelcomponent == 'onetransit':
            prior_d['period'] = 8.32483
            prior_d['t0'] = 1574.27273
            # prior_d['r'] = rp_rs
            prior_d['log_r'] = np.log(rp_rs)
            prior_d['b'] = 0.5  # initialize for broad prior
            prior_d['u[0]'] = 0.3249
            prior_d['u[1]'] = 0.235
            prior_d['r_star'] = RSTAR
            prior_d['logg_star'] = LOGG
            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                if name == 'tess':
                    raise NotImplementedError
                # model [per ground-transit] is :
                # a0+ a1*(time-midtime) + a2*(time-midtime)^2.
                # a0 is the mean, already above.
                prior_d[f'{name}_mean'] = 1
                prior_d[f'{name}_a1'] = 0
                prior_d[f'{name}_a2'] = 0

        if modelcomponent == 'allindivtransit':
            prior_d['period'] = P_orb
            prior_d['t0'] = t0_orb
            prior_d['log_r'] = np.log(rp_rs)
            prior_d['b'] = 0.95

            prior_d['r_star'] = RSTAR
            prior_d['logg_star'] = LOGG

            # T-band Teff 6000K, logg 4.50 (Claret+18)
            # prior_d['u[0]'] = 0.3249
            # prior_d['u[1]'] = 0.235
            prior_d['u'] = [0.3249, 0.235]

            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                # mean + a1*(time-midtime) + a2*(time-midtime)^2.
                prior_d[f'{name}_mean'] = 1
                prior_d[f'{name}_a1'] = 0
                prior_d[f'{name}_a2'] = 0

        if 'rv' in modelcomponent:
            raise NotImplementedError

        if 'gp' in modelcomponent:
            raise NotImplementedError

        if 'sincos' in modelcomponent:
            raise NotImplementedError('try billy')

    return prior_d
