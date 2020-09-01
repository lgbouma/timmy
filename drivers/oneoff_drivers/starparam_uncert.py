# PARSEC:
# Mstar = 1.06 Msun, +/- 0.03
# Teff = 5890, +/- 50
# logg = 4.515, +/- 0.020
# Rstar = 0.940 +/- 0.025 Rsun
# rhostar = 1.80 +/- 0.05 g/cc
# 
# MIST:
# Mstar = 1.118 +/- 0.011
# Teff = 6047 +/- 40
# logg = 4.467 +/- 0.010
# Rstar = 1.022 +/- 0.015
# rhostar = (post-transit-fit...) 1.47 +/- 0.04 g/cc

import numpy as np

d_parsec = {
    'mstar': (1.06, 0.03),
    'teff': (5890, 50),
    'logg': (4.515, 0.02),
    'rstar': (0.940, 0.025)
}

d_mist = {
    'mstar': (1.118, 0.011),
    'teff': (6047, 40),
    'logg': (4.467, 0.010),
    'rstar': (1.022, 0.015)
}

params = ['mstar', 'teff', 'logg', 'rstar']

for p in params:

    unc_sys = (d_mist[p][0] - d_parsec[p][0])
    unc_stat = d_mist[p][1]

    val = np.average([d_mist[p][0], d_parsec[p][0]])
    unc = np.sqrt(unc_sys**2 + unc_stat**2)

    print(f'AVERAGE {p}: {val} +/- {unc}')
    print(f'MIST {p}: {d_mist[p][0]} +/- {unc}')


