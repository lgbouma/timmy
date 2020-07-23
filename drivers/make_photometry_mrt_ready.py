"""
Make the photometry table.
"""

import os
import numpy as np, pandas as pd
from numpy import array as nparr
from collections import OrderedDict
from timmy.convenience import (
    get_clean_tessphot, get_elsauce_phot, _subset_cut,
    get_astep_phot
)

provenance = 'spoc' # could be "cdips"
yval = 'PDCSAP_FLUX' # could be SAP_FLUX 
x_obs, y_obs, y_err = get_clean_tessphot(provenance, yval, binsize=None,
                                         maskflares=1)

datasets = OrderedDict()
# datasets['tess'] = [x_obs, y_obs, y_err]

datestrs = ['20200401', '20200426', '20200521', '20200614']
for ix, d in enumerate(datestrs):
    x_obs, y_obs, y_err = get_elsauce_phot(datestr=d)
    x_obs -= 2457000 # convert to BTJD
    datasets[f'elsauce_{ix}'] = [x_obs, y_obs, y_err]

datestrs = ['20200529', '20200614', '20200623']
for ix, d in enumerate(datestrs):
    x_obs, y_obs, y_err = get_astep_phot(datestr=d)
    x_obs += 2450000 # convert to BJD_TDB
    x_obs -= 2457000 # convert to BTJD
    datasets[f'astep_{ix}'] = [x_obs, y_obs, y_err]

times, fluxs, errs, instrs = nparr([]), nparr([]), nparr([]), nparr([])
for k in datasets.keys():
    times = np.hstack((times, datasets[k][0]))
    fluxs = np.hstack((fluxs, datasets[k][1]))
    errs = np.hstack((errs, datasets[k][2]))
    instrs = np.hstack((instrs, np.repeat(k, len(datasets[k][0]))))

df = pd.DataFrame({
    'btjd_tdb': np.round(times,8),
    'flux': np.round(fluxs,6),
    'fluxerr': np.round(errs,6),
    'instr': instrs
})

outpath = '../data/phot/photometry_mrt_ready.csv'
df.to_csv(outpath, index=False)
print(outpath)

import IPython; IPython.embed()
