# Example Keplerian fit configuration file

# Required packages for setup
import os
import pandas as pd
import numpy as np
import radvel

# Define global planetary system and dataset parameters
starname = 'TOI837'
nplanets = 1                                   # number of planets in the system
instnames = ['CHIRON', 'FEROS']                # list of instrument names. Can be whatever you like (no spaces) but should match 'tel' column in the input file.
ntels = len(instnames)                         # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw logk'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 0                                       # reference epoch for RV timestamps (i.e. this number has been subtracted off your timestamps)
planet_letters = {1: 'b'}                      # map the numbers in the Parameters keys to planet letters (for plotting and tables)

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets,basis='per tc e w k', planet_letters=planet_letters)    # initialize Parameters object

anybasis_params['per1'] = radvel.Parameter(value=8.32467)          # period of 1st planet
anybasis_params['tc1'] = radvel.Parameter(value=1574.2738+2457000) # time of inferior conjunction (transit) of 1st planet
anybasis_params['e1'] = radvel.Parameter(value=0.1)                # eccentricity of 1st planet
anybasis_params['w1'] = radvel.Parameter(value=np.pi/2.)           # argument of periastron of the star's orbit for 1st planet
anybasis_params['k1'] = radvel.Parameter(value=50)                 # velocity semi-amplitude for 1st planet

time_base = 2458800                                          # abscissa for slope and curvature terms (should be near mid-point of time baseline)
anybasis_params['dvdt'] = radvel.Parameter(value=0.0)        # slope: (If rv is m/s and time is days then [dvdt] is m/s/day)
anybasis_params['curv'] = radvel.Parameter(value=0.0)        # curvature: (If rv is m/s and time is days then [curv] is m/s/day^2)

anybasis_params['gamma_CHIRON'] = radvel.Parameter(value=0.0)     # velocity zero-point
anybasis_params['gamma_FEROS'] = radvel.Parameter(value=0.0)      # "

anybasis_params['jit_CHIRON'] = radvel.Parameter(value=10)        # jitter
anybasis_params['jit_FEROS'] = radvel.Parameter(value=10)         # "

# Convert input orbital parameters into the fitting basis
params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

# Set the 'vary' attributes of each of the parameters in the fitting basis. A parameter's 'vary' attribute should
# be set to False if you wish to hold it fixed during the fitting process. By default, all 'vary' parameters
# are set to True.

params['secosw1'].vary = True
params['sesinw1'].vary = True
params['per1'].vary = True  # if false, struggles more w/ convergence.
params['tc1'].vary = True

params['curv'].vary = False
params['dvdt'].vary = False

# Load radial velocity data, in this example the data is contained in
# an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
# the velocities are expected to be in m/s
from timmy.paths import DATADIR
datestr = '20200525'
cleanrvpath = os.path.join(DATADIR, 'spectra', 'RVs_{}_clean.csv'.format(datestr))
data = pd.read_csv(cleanrvpath)

# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.HardBounds('logk1', np.log(0.1), np.log(1e5)), # Positive K Prior recommended
    # radvel.prior.PositiveKPrior( nplanets ),
    radvel.prior.Gaussian('per1', params['per1'].value, 5e-3),
    radvel.prior.Gaussian('tc1', params['tc1'].value, 1e-2),
    radvel.prior.HardBounds('jit_FEROS', 0.0, 200),
    radvel.prior.HardBounds('jit_CHIRON', 0.0, 200)
]

# optional argument that can contain stellar mass in solar units (mstar) and
# uncertainty (mstar_err). If not set, mstar will be set to nan.
stellar = dict(mstar=1.1, mstar_err=0.2)
planet = dict(rp1=1.0, rperr1=0.2)
