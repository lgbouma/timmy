import os
import timmy.plotting as tp
from timmy.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cluster_membership')

for color0 in ['phot_bp_mean_mag', 'phot_g_mean_mag']:
    tp.plot_hr(PLOTDIR, isochrone=1, color0=color0)
    tp.plot_hr(PLOTDIR, isochrone=0, color0=color0)
    tp.plot_hr(PLOTDIR, isochrone=1, do_cmd=1, color0=color0)
    tp.plot_hr(PLOTDIR, isochrone=0, do_cmd=1, color0=color0)
