"""
Go from the RVs Christoph Bergmann sent (with delta Pav as the
template) to RVs that can be input to radvel.
"""
import os
import pandas as pd, numpy as np
from astrobase.lcmath import find_lc_timegroups
from numpy import array as nparr

from timmy.paths import DATADIR

rvdir = os.path.join(DATADIR, 'spectra', 'Veloce', 'RVs')
rvpath = os.path.join(rvdir, 'TOI837_rvs_v1.txt')

df = pd.read_csv(rvpath, names=['time','rv','rv_err'], sep=' ')

ngroups, groupinds = find_lc_timegroups(nparr(df.time), mingap=0.8)

times, rvs, rverrs = [], [], []
for g in groupinds:
    times.append( df.loc[g].time.mean() )
    rvs.append( df.loc[g].rv.mean() )
    rverrs.append( df.loc[g].rv.std() )

veloce_df = pd.DataFrame({
    'time': times,
    'mnvel': nparr(rvs) - np.nanmean(rvs),
    'errvel': rverrs
})
veloce_df['tel'] = 'Veloce'
veloce_df['Name'] = 'toi837'
veloce_df['Source'] = 'Bergmann'

old_df = pd.read_csv(
    os.path.join(DATADIR, 'spectra', 'RVs_20200525_clean.csv')
)

new_df = pd.concat((old_df, veloce_df))

outpath = os.path.join(DATADIR, 'spectra', 'RVs_20200624_clean.csv')
new_df.to_csv(outpath, index=False)
print(f'made {outpath}')
