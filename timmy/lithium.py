"""
Tools to parse lithium results from other people.
"""

import numpy as np, pandas as pd
import os
from astropy.io import fits
from astropy.table import Table

from timmy.paths import DATADIR
from glob import glob

lithiumdir = os.path.join(DATADIR, 'lithium')

def get_Randich18_lithium():

    dfpath = os.path.join(
        lithiumdir, 'Randich_2018_Gaia-ESO_7_clusters_plottable.csv'
    )

    if not os.path.exists(dfpath):
        hl = fits.open(
            os.path.join(lithiumdir, 'Randich_2018_Gaia-ESO_7_clusters.fits')
        )

        ts = []
        for i in range(1,8):
            ts.append(Table(hl[i].data).to_pandas())

        df = pd.concat(ts)

        outpath = dfpath.replace('plottable.csv','full.csv')
        df.to_csv(outpath, index=False)

        sel = (df.MembPA > 0.9) | (df.MembPB > 0.9)

        sdf = df[sel]
        sdf.to_csv(dfpath, index=False)

    return pd.read_csv(dfpath)


def get_Berger18_lithium():

    hl = fits.open(
        os.path.join(lithiumdir, 'Berger_2018_table1.fits')
    )

    df = Table(hl[1].data).to_pandas()

    return df
