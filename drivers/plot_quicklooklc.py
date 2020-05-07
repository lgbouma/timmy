import os
from timmy.paths import DATADIR, RESULTSDIR
from timmy.plotting import plot_quicklooklc

overwrite = 1

outdir = os.path.join(RESULTSDIR, 'hacky_plots')

plot_quicklooklc(outdir, yval='SAP_FLUX', overwrite=overwrite)
plot_quicklooklc(outdir, yval='PDCSAP_FLUX', overwrite=overwrite)

for ap in range(1,4):
    plot_quicklooklc(outdir, yval='IRM{}'.format(ap), provenance='cdips',
                     overwrite=overwrite)
    plot_quicklooklc(outdir, yval='PCA{}'.format(ap), provenance='cdips',
                     overwrite=overwrite)

