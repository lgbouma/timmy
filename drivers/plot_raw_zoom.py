import os
from timmy.paths import DATADIR, RESULTSDIR
from timmy.plotting import plot_raw_zoom

overwrite = 0
outdir = os.path.join(RESULTSDIR, 'paper_plots')

for yval in ['PDCSAP_FLUX', 'SAP_FLUX']:
    for detrend in [0,1]:
        plot_raw_zoom(outdir, yval=yval, detrend=detrend, overwrite=overwrite)
        plot_raw_zoom(outdir, yval=yval, detrend=detrend, overwrite=overwrite)
