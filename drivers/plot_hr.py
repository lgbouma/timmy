import os
import timmy.plotting as tp
from timmy.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cluster_membership')

tp.plot_hr(PLOTDIR, isochrone=1, do_cmd=1)
tp.plot_hr(PLOTDIR, isochrone=0, do_cmd=1)
tp.plot_hr(PLOTDIR, isochrone=1)
tp.plot_hr(PLOTDIR, isochrone=0)
