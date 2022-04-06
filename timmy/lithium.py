"""
Tools to parse lithium results from other people.

    get_Kraus14_Mentuch08_TucHor
    get_Randich01_lithium
    get_Randich18_lithium
    get_Berger18_lithium
"""

import numpy as np, pandas as pd
import os
from astropy.io import fits
from astropy.table import Table

from timmy.paths import DATADIR
from glob import glob

lithiumdir = os.path.join(DATADIR, 'lithium')

def get_Kraus14_Mentuch08_TucHor():

    # from Trevor's compilation
    csvpath = os.path.join(lithiumdir, 'TucHor_Kraus2014_Mentuch08.csv')
    df = pd.read_csv(csvpath, delim_whitespace=True)

    df = df.rename({
        'Teff':'Teff',
        'WLi_limit':'f_EWLi',
        'WLi':'EWLi',
        'e_WLi':'e_EWLi'
    }, axis='columns')
    df['e_Teff'] = np.nan

    # impose uncertainties. these are mostly the randich cases.
    sel = ( (df.Teff < 4500) & (pd.isnull(df.e_EWLi)) & (df.f_EWLi != '1') )
    df.loc[sel, 'e_EWLi'] = 40
    sel = ( (df.Teff > 4500) & (pd.isnull(df.e_EWLi)) & (df.f_EWLi != '1') )
    df.loc[sel, 'e_EWLi'] = 20

    return df

def get_Randich01_lithium():

    randich01paths = glob(os.path.join(
        lithiumdir, "Randich_2001_AA_372_862*.fits"
    ))
    assert len(randich01paths) == 3

    names, teffs, e_Teffs, limits, EW_Lis, e_EW_Lis = (
        [],[],[],[],[],[]
    )

    for randich01path in randich01paths:
        hl = fits.open(randich01path)
        d = hl[1].data
        hl.close()

        names.append(d['Name'])
        teffs.append(d['Teff'])

        # Table 4 has different formatting, including giving EW Li in 0.1pm,
        # rather than pm.  (Both of which are silly units).
        if 'table4' not in randich01path:
            e_Teffs.append(d['e_Teff'])
            limit_key = 'l_EW_Li_'
            Li_key = 'EW_Li_'
            err_key = 'e_EW_Li_'
            mfactor = 10
            e_EW_Lis.append(mfactor*d[err_key])
        else:
            e_Teffs.append(np.nan*np.ones(len(d)))
            limit_key = 'l_EW_li_R97'
            Li_key = 'EW_li_'
            mfactor = 100
            e_EW_Lis.append(np.nan*np.ones(len(d)))

        limits.append(d[limit_key])
        EW_Lis.append(mfactor*d[Li_key])

    df = pd.DataFrame({
        'name': np.hstack(names),
        'Teff': np.hstack(teffs),
        'e_Teff': np.hstack(e_Teffs),
        'f_EWLi': np.hstack(limits),
        'EWLi': np.hstack(EW_Lis),
        'e_EWLi': np.hstack(e_EW_Lis)
    })

    # impose uncertainties. these are mostly the randich cases.
    sel = ( (df.Teff < 4500) & (pd.isnull(df.e_EWLi)) & (df.f_EWLi != '<=') )
    df.loc[sel, 'e_EWLi'] = 40
    sel = ( (df.Teff > 4500) & (pd.isnull(df.e_EWLi)) & (df.f_EWLi != '<=') )
    df.loc[sel, 'e_EWLi'] = 20

    return df

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
