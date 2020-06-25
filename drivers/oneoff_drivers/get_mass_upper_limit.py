import os
import pandas as pd, numpy as np
from astrobase.lcmath import find_lc_timegroups
from numpy import array as nparr
from astropy import units as u

from timmy.paths import DATADIR, RESULTSDIR

datestr = '20200624'
postpath = os.path.join(RESULTSDIR, 'radvel_fitting',
                        f'{datestr}_simple_planet',
                        'TOI837_derived.csv.tar.bz2')

df = pd.read_csv(postpath)

cutoff = 99.7 # 3-sigma

val = np.percentile(nparr(df.mpsini1), cutoff)*u.Mearth

print(f'Mpsini 3sigma (99.7th percentile): {val}')
print(f'Mpsini 3sigma (99.7th percentile): {val.to(u.Mjup)}')

postpath = os.path.join(RESULTSDIR, 'radvel_fitting',
                        f'{datestr}_simple_planet',
                        'TOI837_chains.csv.tar.bz2')
df = pd.read_csv(postpath)

val = np.percentile(nparr(df.logk1), cutoff)

print(f'logk1 3sigma (99.7th percentile): {val:.8f}')
