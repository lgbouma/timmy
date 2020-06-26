import pandas as pd, numpy as np
import os
from timmy.rotationperiod import measure_rotation_period_and_unc
from timmy.convenience import (
    get_clean_tessphot, detrend_tessphot, get_elsauce_phot, _subset_cut
)
from timmy.paths import RESULTSDIR

outpath = os.path.join(RESULTSDIR, 'paper_tables', 'rotation_period.csv')

# get data
provenance = 'spoc'
yval = 'PDCSAP_FLUX'
x_obs, y_obs, y_err = get_clean_tessphot(provenance, yval, binsize=None,
                                         maskflares=1)
s = np.isfinite(y_obs) & np.isfinite(x_obs) & np.isfinite(y_err)
x_obs, y_obs, y_err = x_obs[s], y_obs[s], y_err[s]

plotpath = os.path.join(RESULTSDIR, 'paper_tables', 'rotation_period_check.png')

period, period_unc = measure_rotation_period_and_unc(
    x_obs, y_obs, period_min=1, period_max=10, period_fit_cut=0.5, nterms=1,
    plotpath=plotpath
)

df = pd.DataFrame({'period':period, 'period_unc':period_unc}, index=[0])

df.to_csv(outpath, index=False)

# 1 term: 2.987 +/- 0.056
# 2 terms: 3.004 +/- 0.053
