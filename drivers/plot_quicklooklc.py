import os
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt

from timmy.paths import DATADIR, RESULTSDIR
from timmy.plotting import plot_quicklooklc

overwrite = 1

outdir = os.path.join(RESULTSDIR, 'hacky_plots')

plot_quicklooklc(outdir, yval='SAP_FLUX', overwrite=overwrite)
plot_quicklooklc(outdir, yval='PDCSAP_FLUX', overwrite=overwrite)
plot_quicklooklc(outdir, yval='IRM1', provenance='cdips', overwrite=overwrite)
plot_quicklooklc(outdir, yval='PCA1', provenance='cdips', overwrite=overwrite)
