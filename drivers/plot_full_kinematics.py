import os
import timmy.plotting as tp
from timmy.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cluster_membership')

tp.plot_full_kinematics(PLOTDIR)
