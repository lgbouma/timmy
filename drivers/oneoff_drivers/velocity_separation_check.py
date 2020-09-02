import os
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy import units as u, constants as const
from aesthetic.plot import set_style, savefig

from timmy.priors import RSTAR, MSTAR

sma = np.logspace(-1, 3, 100)*u.AU

Mtot = (MSTAR)*u.Msun

v_kep = ((const.G * Mtot / sma)**(1/2)).to(u.km/u.s)

set_style()

f,ax = plt.subplots()
ax.plot(sma, v_kep, label=f'Mtot={MSTAR:.2f}M$_\odot$')
ax.plot(sma, v_kep*np.sqrt(2), label=f'Mtot={2*MSTAR:.2f}M$_\odot$')

ax.set_xlabel('semi-major axis [AU]')
ax.set_ylabel('$v_{\mathrm{Kep}}$ [km/s]')
ax.set_xscale('log')
ax.set_yscale('log')

ax.hlines([10,15], 1e-1, 1e3, label='SB2 sep limits')
ax.set_xlim((1e-1, 1e3))

ax.legend()

figpath = '../../results/fpscenarios/velocity_separation_check.png'
savefig(f, figpath, writepdf=False, dpi=300)

