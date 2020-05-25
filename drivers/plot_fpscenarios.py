import os
import timmy.plotting as tp
from timmy.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'fpscenarios')

tp.plot_fpscenarios(PLOTDIR)
