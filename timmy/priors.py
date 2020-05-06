import numpy as np
from collections import OrderedDict

def initialize_prior_d(modelcomponents):

    # SPOC multisector report
    P_orb = 8.32467 # +/- 4e-4
    t0_orb = 1574.2738  # +/- 1.1e-3,  BTJD
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

        if 'transit' in modelcomponent:
            prior_d['period'] = P_orb
            prior_d['t0'] = t0_orb
            prior_d['r'] = rp_rs
            prior_d['b'] = 0.5  # initialize for broad prior
            prior_d['u'] = [0.3189,0.2278] # Teff 6300K, logg 4.50 (Claret+18)
            prior_d['mean'] = 0

        if 'gp' in modelcomponent:
            prior_d['P_rot'] = P_rot
            prior_d['amp'] = amp
            prior_d['mix'] = amp_mix
            prior_d['log_Q0'] = log_Q0
            prior_d['log_deltaQ'] = log_deltaQ

        if 'sincos' in modelcomponent:

            raise NotImplementedError('try billy')

    return prior_d
