'''
DESCRIPTION
----------
Make scatter plot of olbliquity vs age using Dai+20 results (which presumably
are from TEPCAT).
'''

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter

import pandas as pd, numpy as np
import os
from copy import deepcopy

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

from cdips.utils import today_YYYYMMDD
from aesthetic.plot import savefig, format_ax, set_style
from timmy.paths import DATADIR

def arr(x):
    return np.array(x)

def plot_obliquity_vs_age_scatter(active_targets=1, specialyoung=1,
                             showcandidates=0, show_legend=0):

    set_style()

    df = pd.read_csv(os.path.join(DATADIR, 'age_versus', 'lambda_vs_age.csv'))

    # #
    # # columns described at
    # # https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
    # #
    # ea_tab = NasaExoplanetArchive.query_criteria(
    #     table="exoplanets", select="*", cache=True
    # )

    # #
    # # get systems with finite ages (has a value, and +/- error bar)
    # #
    # has_age_value = ~ea_tab['st_age'].mask
    # has_age_errs  = (~ea_tab['st_ageerr1'].mask) & (~ea_tab['st_ageerr2'].mask)
    # has_rp_value = ~ea_tab['pl_rade'].mask
    # has_rp_errs  = (~ea_tab['pl_radeerr1'].mask) & (~ea_tab['pl_radeerr2'].mask)
    # rp_gt_0 = (ea_tab['pl_rade'] > 0)
    # m_gt_0 = (ea_tab['pl_bmassj'] > 0) & (np.abs(ea_tab['pl_bmassjerr1']) > 0)
    # has_m_errs  = (~ea_tab['pl_bmassjerr1'].mask) & (~ea_tab['pl_bmassjerr2'].mask)

    # transits = (ea_tab['pl_tranflag']==1)

    # sel = (
    #     has_age_value & has_age_errs & has_rp_value & has_rp_errs & transits &
    #     rp_gt_0 & m_gt_0 & has_m_errs
    # )

    # t = ea_tab[sel]
    # tyoung = t[(t['st_age'] < 0.1*u.Gyr) & (t['st_age'] > 0*u.Gyr)]

    #
    # read params
    #
    age = df['age_gyr']
    # age_perr = t['st_ageerr1']
    # age_merr = np.abs(t['st_ageerr2'])
    # age_errs = np.array([age_perr, age_merr]).reshape(2, len(age))

    obliq = df['lambda_deg']
    # rp = t['pl_rade']
    # rp_perr = t['pl_radeerr1']
    # rp_merr = t['pl_radeerr2']
    # rp_errs = np.array([rp_perr, rp_merr]).reshape(2, len(age))
    # # rp /= 11.2089 # used jupiter radii

    # mass = t['pl_bmassj']

    ##########################################
    fig,ax = plt.subplots(figsize=(4,3))

    label = (
        'TEPCAT'
    )

    ax.scatter(age*1e9, obliq, color='darkgray', s=3, zorder=1, marker='o',
               linewidth=0, label=label, alpha=1)

    #
    # targets
    #
    target_age = (np.array([3.5e7])*u.yr).to(u.Gyr)
    target_rp = (np.array([0.77])*u.Rjup).to(u.Rearth)
    target_rp_unc = (np.array([0.07, 0.09])*u.Rjup).to(u.Rearth)[:,None]

    if active_targets:

        label = (
            'TOI$\,$837'
        )

        ax.plot(target_age*1e9, target_mass, mew=0.5, markerfacecolor='yellow',
                markersize=15, marker='*', color='k', lw=0, label=label,
                zorder=3)

    if specialyoung:

        youngnames = df[~pd.isnull(df['comment'])].comment

        markertypes= ['o', 'v', 'X', 's', 'P', 'd']

        for ix, y in enumerate(np.unique(youngnames)):

            s = (df['comment'] == y)

            ax.plot(df[s]['age_gyr']*1e9, df[s]['lambda_deg'], mew=0.5,
                    markerfacecolor='white', markersize=7,
                    marker=markertypes[ix], color='k', lw=0, label=y,
                    zorder=2)

        # # two extra systems...
        # N_uniq = len(np.unique(youngnames))

        # ax.plot(1.5e7/1e9, 10.02, mew=0.5, markerfacecolor='white',
        #         markersize=7, marker=markertypes[N_uniq], color='k', lw=0,
        #         label='HIP 67522', zorder=2)

        # # HD 63433 (TOI 1726, TIC 130181866) omitted -- 400 Myr is old!

    if showcandidates:

        from cdips_followup.manage_candidates import get_candidate_params
        vdf, sdf, _age, _rp, _rp_unc, _period = (
            get_candidate_params(isvalidated=0,
                                 ismanualsubset=1)
        )

        ax.plot(_age*1e9, _rp, mew=0.5, markerfacecolor='lightskyblue', markersize=8,
                marker='*', color='k', lw=0, label='New Planet Candidates',
                zorder=1)

    # flip default legend order
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles[::-1], labels[::-1], loc='upper left',
                        borderpad=0.3, handletextpad=0.5, fontsize=6,
                        framealpha=0)

        leg.get_frame().set_linewidth(0.5)

    ax.set_xlabel('Age [years]')
    ax.set_ylabel(r'Stellar Obliquity [degrees]')
    ax.set_xlim([6e-3*1e9, 17*1e9])
    ax.set_ylim([-20, 200])
    format_ax(ax)
    ax.set_xscale('log')

    savstr = '_no_overplot' if not active_targets else '_toi837'
    if showcandidates:
        savstr += '_showcandidates'
    if show_legend:
        savstr += '_yeslegend'
    else:
        savstr += '_nolegend'

    outpath = (
        '../results/obliquity_vs_age_scatter/obliquity_vs_age_scatter_{}{}.png'.
        format(today_YYYYMMDD(), savstr)
    )

    savefig(fig, outpath, writepdf=1, dpi=400)



if __name__=='__main__':

    for show_legend in [0,1]:
        for specialyoung in [0,1]:
            plot_obliquity_vs_age_scatter(active_targets=0,
                                          specialyoung=1,
                                          show_legend=show_legend)
