"""
Given the RVs, what limits on additional massive companions can be placed?

This driver script assumes drivers.radvel_drivers.TOI837_fpscenario_limits has
been executed.

It must be executed for the "fpscenario" plot to be created.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
from timmy.paths import RESULTSDIR
import corner
from numpy import array as nparr
from timmy.plotting import plot_contour_2d_samples
from astropy import units as u, constants as const

##########################################
# config #
n_grid = int(5e1)       # number of points in logspaced grid
smooth = 2              # gaussian smoothing over of sampling grid
##########################################

chainpath = os.path.join(RESULTSDIR, 'radvel_fitting', '20200525_fpscenario',
                         'TOI837_fpscenario_limits_chains.csv.tar.bz2')

df = pd.read_csv(chainpath)

params = ['logper1', 'logk1']

sdf = df[params]

logper1 = nparr(sdf['logper1'])
logk1 = nparr(sdf['logk1'])
xsample = logper1
ysample = logk1

xgrid = np.linspace(xsample.min(), xsample.max(), num=n_grid)
ygrid = np.linspace(ysample.min(), ysample.max(), num=n_grid)

outpath = '../results/fpscenarios/rvoutersensitivity_logper_logk.png'
plot_contour_2d_samples(xsample, ysample, xgrid, ygrid, outpath,
                        xlabel='logper', ylabel='logk', smooth=smooth)

e = 0
sini = 1
Mstar = 1.1 * u.Msun
# NOTE: assuming log[k (m/s)]
# NOTE: this is the Mstar >> Mcomp approximation. If this doesn't hold, it's an
# implicit equation (the "binary mass function"). So in the large semimajor
# axis regime, your resulting converted contours are wrong.
msini_msun = ((
    (np.exp(logk1) / (28.4329)) * (1-e**2)**(1/2) *
    (Mstar.to(u.Msun).value)**(2/3) *
    ((np.exp(logper1)*u.day).to(u.yr).value)**(1/3)
)*u.Mjup).to(u.Msun).value

logmsini = np.log(msini_msun)

per1 = np.exp(logper1)*u.day
sma1 = ((
    per1**2 * const.G*(Mstar + msini_msun*u.Msun) / (4*np.pi**2)
)**(1/3)).to(u.au).value

logsma1 = np.log(sma1)


xsample = np.log10(np.exp(logsma1))
ysample = np.log10(np.exp(logmsini))

xgrid = np.linspace(xsample.min(), xsample.max(), num=n_grid)
ygrid = np.linspace(ysample.min(), ysample.max(), num=n_grid)

outpath = '../results/fpscenarios/rvoutersensitivity_log10sma_log10msini.png'
x_3sig, y_3sig = plot_contour_2d_samples(xsample, ysample, xgrid, ygrid,
                                         outpath, xlabel='log10sma',
                                         ylabel='log10mpsini',
                                         return_3sigma=True, smooth=smooth)

# NB these vertices are in log10space
sel = (x_3sig < 5.5) & (x_3sig > 0) & (y_3sig > -3.8) & (y_3sig < 3)

outdf = pd.DataFrame({
    'log10sma': x_3sig[sel],
    'log10mpsini': y_3sig[sel]
})
outdf = outdf.sort_values(by='log10sma')

outpath = '../results/fpscenarios/rvoutersensitivity_3sigma.csv'

outdf.to_csv(outpath, index=False)
print(f'made {outpath}')
