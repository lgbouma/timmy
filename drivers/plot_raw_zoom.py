import os
from timmy.paths import DATADIR, RESULTSDIR
from timmy.plotting import plot_raw_zoom

overwrite = 1
outdir = os.path.join(RESULTSDIR, 'paper_plots')

plot_raw_zoom(outdir, yval='PDCSAP_FLUX', overwrite=overwrite)
