"""
Plots:

    plot_quicklooklc
    plot_raw_zoom
    plot_phasefold
    plot_scene
    plot_hr
    (plot_positions)
    (plot_velocities)
    plot_full_kinematics

    groundphot:
        plot_groundscene
        shift_img_plot
        plot_pixel_lc
        vis_photutils_lcs
        stackviz_blend_check
"""
import os, corner, pickle
from datetime import datetime
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from numpy import array as nparr
from itertools import product

from billy.plotting import savefig, format_ax
import billy.plotting as bp

from timmy.paths import DATADIR, RESULTSDIR
from timmy.convenience import get_data, get_clean_data, detrend_data

from astrobase.lcmath import (
    phase_magseries, phase_bin_magseries, sigclip_magseries,
    find_lc_timegroups
)
from astrobase import periodbase
from astrobase.plotbase import skyview_stamp

from astropy.stats import LombScargle
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.time import Time

from astropy.wcs import WCS
from astroquery.mast import Catalogs
import astropy.visualization as vis
import matplotlib as mpl
from matplotlib import patches

##################################################
# wrappers to generic plots implemented in billy #
##################################################

def plot_test_data(x_obs, y_obs, y_mod, modelid, outdir):
    bp.plot_test_data(x_obs, y_obs, y_mod, modelid, outdir)


def plot_MAP_data(x_obs, y_obs, y_MAP, outpath):
    bp.plot_MAP_data(x_obs, y_obs, y_MAP, outpath, ms=1)


def plot_sampleplot(m, outpath, N_samples=100):
    bp.plot_sampleplot(m, outpath, N_samples=N_samples, ms=1, malpha=0.1)


def plot_traceplot(m, outpath):
    bp.plot_traceplot(m, outpath)


def plot_cornerplot(true_d, m, outpath):
    bp.plot_cornerplot(true_d, m, outpath)

##################
# timmy-specific #
##################

def plot_splitsignal_map(m, outpath):
    """
    y_obs + y_MAP + y_rot + y_orb
    things at rotation frequency
    things at orbital frequency
    """

    plt.close('all')
    # 8.5x11 is letter paper. x10 allows space for caption.
    fig, axs = plt.subplots(nrows=4, figsize=(8.5, 10), sharex=True)

    axs[0].set_ylabel('Raw flux', fontsize='x-large')
    axs[0].plot(m.x_obs, m.y_obs, ".k", ms=4, label="data", zorder=2,
                rasterized=True)
    axs[0].plot(m.x_obs, m.map_estimate['mu_model'], lw=0.5, label='MAP',
                color='C0', alpha=1, zorder=1)

    y_tra = m.map_estimate['mu_transit']
    y_gprot = m.map_estimate['mu_gprot']

    axs[1].set_ylabel('Transit', fontsize='x-large')
    axs[1].plot(m.x_obs, m.y_obs-y_gprot, ".k", ms=4, label="data-rot",
                zorder=2, rasterized=True)
    axs[1].plot(m.x_obs, m.map_estimate['mu_model']-y_gprot, lw=0.5,
                label='model-rot', color='C0', alpha=1, zorder=1)

    axs[2].set_ylabel('Rotation', fontsize='x-large')
    axs[2].plot(m.x_obs, m.y_obs-y_tra, ".k", ms=4, label="data-transit",
                zorder=2, rasterized=True)
    axs[2].plot(m.x_obs, m.map_estimate['mu_model']-y_tra, lw=0.5,
                label='model-transit', color='C0', alpha=1, zorder=1)

    axs[3].set_ylabel('Residual', fontsize='x-large')
    axs[3].plot(m.x_obs, m.y_obs-m.map_estimate['mu_model'], ".k", ms=4,
                label="data", zorder=2, rasterized=True)
    axs[3].plot(m.x_obs, m.map_estimate['mu_model']-m.map_estimate['mu_model'],
                lw=0.5, label='model', color='C0', alpha=1, zorder=1)


    axs[-1].set_xlabel("Time [days]", fontsize='x-large')
    for a in axs:
        format_ax(a)
        # a.set_ylim((-.075, .075))
        # if part == 'i':
        #     a.set_xlim((0, 9))
        # else:
        #     a.set_xlim((10, 20.3))

    # props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
    #              linewidth=0)
    # if part == 'i':
    #     axs[3].text(0.97, 0.03, 'Orbit 19', ha='right', va='bottom',
    #                 transform=axs[3].transAxes, bbox=props, zorder=3,
    #                 fontsize='x-large')
    # else:
    #     axs[3].text(0.97, 0.03, 'Orbit 20', ha='right', va='bottom',
    #                 transform=axs[3].transAxes, bbox=props, zorder=3,
    #                 fontsize='x-large')

    fig.tight_layout(h_pad=0., w_pad=0.)

    if not os.path.exists(outpath) or m.OVERWRITE:
        savefig(fig, outpath, writepdf=1, dpi=300)

    ydict = {
        'x_obs': m.x_obs,
        'y_obs': m.y_obs,
        'y_resid': m.y_obs-m.map_estimate['mu_model'],
        'y_mod_tra': y_tra,
        'y_mod_gprot': y_gprot,
        'y_mod': m.map_estimate['mu_model'],
        'y_err': m.y_err
    }
    return ydict



def plot_quicklooklc(outdir, yval='PDCSAP_FLUX', provenance='spoc',
                     overwrite=0):

    outpath = os.path.join(
        outdir, 'quicklooklc_{}_{}.png'.format(provenance, yval)
    )

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    # NOTE: I checked that pre-quality cuts etc, there weren't extra transits
    # sitting in the SPOC data.

    time, flux, flux_err = get_clean_data(provenance, yval, binsize=None)

    from wotan import flatten
    # flat_flux, trend_flux = flatten(time, flux, method='pspline',
    #                                 break_tolerance=0.4, return_trend=True)
    flat_flux, trend_flux = flatten(time, flux, method='hspline',
                                    window_length=0.3,
                                    break_tolerance=0.4, return_trend=True)
    # flat_flux, trend_flux = flatten(time, flux, method='biweight',
    #                                 window_length=0.3, edge_cutoff=0.5,
    #                                 break_tolerance=0.4, return_trend=True,
    #                                 cval=2.0)

    _plot_quicklooklc(outpath, time, flux, flux_err, flat_flux, trend_flux,
                      showvlines=0, provenance=provenance)


def _plot_quicklooklc(outpath, time, flux, flux_err, flat_flux, trend_flux,
                      showvlines=0, figsize=(25,8), provenance=None, timepad=1,
                      titlestr=None):

    t0 = 1574.2738
    per = 8.32467
    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    plt.close('all')

    f,axs = plt.subplots(figsize=figsize, nrows=2, sharex=True)

    xmin, xmax = np.nanmin(time)-timepad, np.nanmax(time)+timepad

    s = 1.5 if provenance == 'spoc' else 2.5*10
    axs[0].scatter(time, flux, c='k', zorder=3, s=s, rasterized=True,
                   linewidths=0)
    axs[0].plot(time, trend_flux, c='C0', zorder=4, rasterized=True, lw=1)

    axs[1].scatter(time, flat_flux, c='k', zorder=3, s=s, rasterized=True,
                   linewidths=0)
    axs[1].plot(time, trend_flux-trend_flux, c='C0', zorder=4,
                rasterized=True, lw=1)

    ymin, ymax = np.nanmin(flux), np.nanmax(flux)
    axs[0].set_ylim((ymin, ymax))
    axs[1].set_ylim(( np.nanmean(flat_flux) - 6*np.nanstd(flat_flux),
                      np.nanmean(flat_flux) + 4*np.nanstd(flat_flux)  ))

    axs[0].set_ylabel('raw')
    axs[1].set_ylabel('detrended')
    axs[1].set_xlabel('BJDTDB')
    if isinstance(titlestr, str):
        axs[0].set_title(titlestr)

    for ax in axs:

        if showvlines and provenance == 'spoc':
            ymin, ymax = ax.get_ylim()

            ax.vlines(
                tra_times, ymin, ymax, colors='C1', alpha=0.5,
                linestyles='--', zorder=-2, linewidths=0.5
            )

            ax.set_ylim((ymin, ymax))

        if showvlines and provenance == 'Evans_2020-04-01':
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                [2458940.537, 2458940.617],
                ymin, ymax, colors='C1', alpha=0.5, linestyles='--', zorder=-2,
                linewidths=0.5
            )
            ax.set_ylim((ymin, ymax))

        if showvlines and provenance == 'Evans_2020-04-26':
            # using SG1 updated ephem from before...
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                [2458940.537+3*8.3252539, 2458940.617+3*8.3252539],
                ymin, ymax, colors='C1', alpha=0.5, linestyles='--', zorder=-2,
                linewidths=0.5
            )
            ax.set_ylim((ymin, ymax))

        ax.set_xlim((xmin, xmax))

        format_ax(ax)

    f.tight_layout(h_pad=0., w_pad=0.)
    savefig(f, outpath, writepdf=0, dpi=300)


def plot_raw_zoom(outdir, yval='PDCSAP_FLUX', provenance='spoc',
                  overwrite=0, detrend=0):

    outpath = os.path.join(
        outdir, 'raw_zoom_{}_{}.png'.format(provenance, yval)
    )
    if detrend:
        outpath = os.path.join(
            outdir, 'raw_zoom_{}_{}_detrended.png'.format(provenance, yval)
        )

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    time, flux, flux_err = get_clean_data(provenance, yval, binsize=None)
    flat_flux, trend_flux = detrend_data(time, flux, flux_err)
    if detrend:
        flux = flat_flux

    t_offset = np.nanmin(time)
    time -= t_offset

    t0 = 1574.2738 - t_offset
    per = 8.32467
    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    plt.close('all')

    ##########################################

    # figsize=(8.5, 10) full page... 10 leaves space.
    fig = plt.figure(figsize=(8.5, 6))

    ax0 = plt.subplot2grid(shape=(2,5), loc=(0,0), colspan=5)

    ax1 = plt.subplot2grid((2,5), (1,0), colspan=1)
    ax2 = plt.subplot2grid((2,5), (1,1), colspan=1)
    ax3 = plt.subplot2grid((2,5), (1,2), colspan=1)
    ax4 = plt.subplot2grid((2,5), (1,3), colspan=1)
    ax5 = plt.subplot2grid((2,5), (1,4), colspan=1)

    all_axs = [ax0,ax1,ax2,ax3,ax4,ax5]
    tra_axs = [ax1,ax2,ax3,ax4,ax5]
    tra_ixs = [0,2,3,4,5]

    # main lightcurve
    yval = (flux - np.nanmean(flux))*1e3
    ax0.scatter(time, yval, c='k', zorder=3, s=0.75, rasterized=True,
                linewidths=0)
    ax0.set_ylim((-20, 20)) # omitting like 1 upper point from the big flare at time 38
    ymin, ymax = ax0.get_ylim()
    ax0.vlines(
        tra_times, ymin, ymax, colors='C1', alpha=0.5,
        linestyles='--', zorder=-2, linewidths=0.5
    )
    ax0.set_ylim((ymin, ymax))
    ax0.set_xlim((np.nanmin(time)-1, np.nanmax(time)+1))

    # zoom-in of raw transits
    for ax, tra_ix in zip(tra_axs, tra_ixs):

        mid_time = t0 + per*tra_ix
        tdur = 2/24. # roughly, in units of days
        n = 3.5
        start_time = mid_time - n*tdur
        end_time = mid_time + n*tdur

        s = (time > start_time) & (time < end_time)

        yval = (flux[s] - np.nanmean(flux[s]))*1e3

        ax.scatter(time[s], yval, c='k', zorder=3, s=7,
                   rasterized=True, linewidths=0)
        ax.set_xlim((start_time, end_time))
        ax.set_ylim((-8, 8))

        # ymin, ymax = ax.get_ylim()
        # ax.vlines(
        #     mid_time, ymin, ymax, colors='C1', alpha=0.5,
        #     linestyles='--', zorder=-2, linewidths=0.5
        # )
        # ax.set_ylim((ymin, ymax))


        if tra_ix > 0:
            # hide the ytick labels
            labels = [item.get_text() for item in
                      ax.get_yticklabels()]
            empty_string_labels = ['']*len(labels)
            ax.set_yticklabels(empty_string_labels)

    for ax in all_axs:
        format_ax(ax)

    fig.text(0.5,-0.01, 'Time [days]', ha='center', fontsize='x-large')
    fig.text(-0.01,0.5, 'Relative flux [part per thousand]', va='center',
             rotation=90, fontsize='x-large')

    fig.tight_layout(h_pad=0.2, w_pad=-1.0)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_phasefold(m, outpath, overwrite=0):

    d = {
        'x_obs': m.x_obs,
        'y_obs': m.y_obs,
        'y_orb': m.y_obs, # NOTE: "detrended" beforehand (else would need to substract stellar variability here)
        'y_resid': m.y_obs-m.map_estimate['mu_model'],
        'y_mod': m.map_estimate['mu_model'],
        'y_err': m.y_err
    }

    P_orb = float(np.nanmedian(m.trace['period']))
    t0_orb = float(np.nanmedian(m.trace['t0']))

    # phase and bin them.
    orb_d = phase_magseries(
        d['x_obs'], d['y_orb'], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=5e-4, minbinelems=3
    )

    # get the samples. shape: N_samples x N_time
    np.random.seed(42)
    N_samples = 100
    y_mod_samples = (
        m.trace.mu_model[
            np.random.choice(
                m.trace.mu_model.shape[0], N_samples, replace=False
            ), :
        ]
    )

    mod_ds = {}
    for i in range(N_samples):
        mod_ds[i] = phase_magseries(
            d['x_obs'], y_mod_samples[i, :], P_orb, t0_orb, wrap=True,
            sort=True
        )

    # make tha plot
    plt.close('all')
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(orb_d['phase']*P_orb*24, orb_d['mags'], color='gray', s=2,
               alpha=0.8, zorder=4, linewidths=0, rasterized=True)
    ax.scatter(orb_bd['binnedphases']*P_orb*24, orb_bd['binnedmags'], color='black',
               s=8, alpha=1, zorder=5, linewidths=0)

    # NOTE: might prefer to do mean model, +/- background band. that looks
    # sicker.
    xvals, yvals = [], []
    for i in range(N_samples):
        xvals.append(mod_ds[i]['phase']*P_orb*24)
        yvals.append(mod_ds[i]['mags'])
        ax.plot(mod_ds[i]['phase']*P_orb*24, mod_ds[i]['mags'], color='C1',
                alpha=0.2, rasterized=True, lw=0.2, zorder=-2)

    # # N_samples x N_times
    # from scipy.ndimage import gaussian_filter1d
    # xvals, yvals = nparr(xvals), nparr(yvals)
    # model_phase = xvals.mean(axis=0)
    # g_std = 100
    # n_std = 2
    # mean = gaussian_filter1d(yvals.mean(axis=0), g_std)
    # diff = gaussian_filter1d(n_std*yvals.std(axis=0), g_std)
    # model_flux_lower = mean - diff
    # model_flux_upper = mean + diff

    # ax.plot(model_phase, model_flux_lower, color='C1',
    #         alpha=0.8, lw=0.5, zorder=3)
    # ax.plot(model_phase, model_flux_upper, color='C1', alpha=0.8,
    #         lw=0.5, zorder=3)
    # ax.fill_between(model_phase, model_flux_lower, model_flux_upper,
    #                 color='C1', alpha=0.5, zorder=3, linewidth=0)

    ax.set_ylabel('Relative flux')
    ax.set_xlabel('Hours from mid-transit')

    ax.set_xlim((-0.035*P_orb*24, 0.035*P_orb*24))
    ax.set_ylim((0.9925, 1.005))

    format_ax(ax)
    fig.tight_layout()

    savefig(fig, outpath, writepdf=1, dpi=300)




def plot_scene(c_obj, img_wcs, img, outpath, Tmag_cutoff=17, showcolorbar=0,
               ap_mask=0, bkgd_mask=0, ticid=None):

    plt.close('all')

    # standard tick formatting fails for these images.
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    #
    # wcs information parsing
    # follow Clara Brasseur's https://github.com/ceb8/tessworkshop_wcs_hack
    # (this is from the CDIPS vetting reports...)
    #
    radius = 6.0*u.arcminute

    nbhr_stars = Catalogs.query_region(
        "{} {}".format(float(c_obj.ra.value), float(c_obj.dec.value)),
        catalog="TIC",
        radius=radius
    )

    try:
        px,py = img_wcs.all_world2pix(
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ra'],
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['dec'],
            0
        )
    except Exception as e:
        print('ERR! wcs all_world2pix got {}'.format(repr(e)))
        raise(e)

    ticids = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ID']
    tmags = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['Tmag']

    sel = (px > 0) & (px < 10) & (py > 0) & (py < 10)
    if isinstance(ticid, str):
        sel &= (ticids != ticid)

    px,py = px[sel], py[sel]
    ticids, tmags = ticids[sel], tmags[sel]

    ra, dec = float(c_obj.ra.value), float(c_obj.dec.value)
    target_x, target_y = img_wcs.all_world2pix(ra,dec,0)

    # geometry: there are TWO coordinate axes. (x,y) and (ra,dec). To get their
    # relative orientations, the WCS and ignoring curvature will usually work.
    shiftra_x, shiftra_y = img_wcs.all_world2pix(ra+1e-4,dec,0)
    shiftdec_x, shiftdec_y = img_wcs.all_world2pix(ra,dec+1e-4,0)

    ###########
    # get DSS #
    ###########
    ra = c_obj.ra.value
    dec = c_obj.dec.value
    sizepix = 220
    try:
        dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                     scaling='Linear', convolvewith=None,
                                     sizepix=sizepix, flip=False,
                                     cachedir='~/.astrobase/stamp-cache',
                                     verbose=True, savewcsheader=True)
    except (OSError, IndexError, TypeError) as e:
        print('downloaded FITS appears to be corrupt, retrying...')
        try:
            dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                         scaling='Linear', convolvewith=None,
                                         sizepix=sizepix, flip=False,
                                         cachedir='~/.astrobase/stamp-cache',
                                         verbose=True, savewcsheader=True,
                                         forcefetch=True)

        except Exception as e:
            print('failed to get DSS stamp ra {} dec {}, error was {}'.
                  format(ra, dec, repr(e)))
            return None, None


    ##########################################

    plt.close('all')
    fig = plt.figure(figsize=(4,9))

    # ax0: TESS
    # ax1: DSS
    ax0 = plt.subplot2grid((2, 1), (0, 0), projection=img_wcs)
    ax1 = plt.subplot2grid((2, 1), (1, 0), projection=WCS(dss_hdr))

    ##########################################

    #
    # ax0: img
    #

    #interval = vis.PercentileInterval(99.99)
    interval = vis.AsymmetricPercentileInterval(5,99)
    vmin,vmax = interval.get_limits(img)
    norm = vis.ImageNormalize(
        vmin=vmin, vmax=vmax, stretch=vis.LogStretch(1000))

    cset0 = ax0.imshow(img, cmap=plt.cm.gray_r, origin='lower', zorder=1,
                       norm=norm)

    if isinstance(ap_mask, np.ndarray):
        for x,y in product(range(10),range(10)):
            if ap_mask[y,x]:
                ax0.add_patch(
                    patches.Rectangle(
                        (x-.5, y-.5), 1, 1, hatch='//', fill=False, snap=False,
                        linewidth=0., zorder=2, alpha=0.7, rasterized=True
                    )
                )

    if isinstance(bkgd_mask, np.ndarray):
        for x,y in product(range(10),range(10)):
            if bkgd_mask[y,x]:
                ax0.add_patch(
                    patches.Rectangle(
                        (x-.5, y-.5), 1, 1, hatch='x', fill=False, snap=False,
                        linewidth=0., zorder=2, alpha=0.7, rasterized=True
                    )
                )

    ax0.scatter(px, py, marker='o', c='C1', s=5e4/(tmags**3), rasterized=True,
                zorder=6, linewidths=0.8)
    # ax0.scatter(px, py, marker='x', c='C1', s=20, rasterized=True,
    #             zorder=6, linewidths=0.8)
    ax0.plot(target_x, target_y, mew=0.5, zorder=5,
             markerfacecolor='yellow', markersize=15, marker='*',
             color='k', lw=0)

    ax0.text(4.2, 5, 'A', fontsize=16, color='C1', zorder=6, style='italic')
    ax0.text(4.6, 4.0, 'B', fontsize=16, color='C1', zorder=6, style='italic')

    ax0.set_title('TESS', fontsize='xx-large')

    if showcolorbar:
        cb0 = fig.colorbar(cset0, ax=ax0, extend='neither', fraction=0.046, pad=0.04)

    #
    # ax1: DSS
    #
    cset1 = ax1.imshow(dss, origin='lower', cmap=plt.cm.gray_r)

    ax1.grid(ls='--', alpha=0.5)
    ax1.set_title('DSS2 Red', fontsize='xx-large')
    if showcolorbar:
        cb1 = fig.colorbar(cset1, ax=ax1, extend='neither', fraction=0.046,
                           pad=0.04)

    # DSS is ~1 arcsecond per pixel. overplot apertures on axes 6,7
    for ix, radius_px in enumerate([21,21*1.5,21*2.25]):
        circle = plt.Circle((sizepix/2, sizepix/2), radius_px,
                            color='C{}'.format(ix), fill=False, zorder=5+ix)
        ax1.add_artist(circle)

    #
    # ITNERMEDIATE SINCE TESS IMAGES NOW PLOTTED
    #
    for ax in [ax0]:
        ax.grid(ls='--', alpha=0.5)
        if shiftra_x - target_x > 0:
            # want RA to increase to the left (almost E)
            ax.invert_xaxis()
        if shiftdec_y - target_y < 0:
            # want DEC to increase up (almost N)
            ax.invert_yaxis()

    for ax in [ax0,ax1]:
        format_ax(ax)
        ax.set_xlabel(r'$\alpha_{2000}$')
        ax.set_ylabel(r'$\delta_{2000}$')

    if showcolorbar:
        fig.tight_layout(h_pad=-8, w_pad=-8)
    else:
        fig.tight_layout(h_pad=1, w_pad=1)

    savefig(fig, outpath, dpi=300)


def plot_hr(outdir):

    # from cdips.tests.test_nbhd_plot
    pklpath = '/Users/luke/Dropbox/proj/timmy/results/cluster_membership/nbhd_info_5251470948229949568.pkl'
    info = pickle.load(open(pklpath, 'rb'))
    (targetname, groupname, group_df_dr2, target_df, nbhd_df,
     cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
     group_in_k13, group_in_cg18, group_in_kc19, group_in_k18
    ) = info

    ##########

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    nbhd_yval = np.array([nbhd_df['phot_g_mean_mag'] +
                          5*np.log10(nbhd_df['parallax']/1e3) + 5])
    ax.scatter(
        nbhd_df['phot_bp_mean_mag']-nbhd_df['phot_rp_mean_mag'], nbhd_yval,
        c='gray', alpha=1., zorder=2, s=5, rasterized=True, linewidths=0,
        label='Neighborhood', marker='.'
    )

    yval = group_df_dr2['phot_g_mean_mag'] + 5*np.log10(group_df_dr2['parallax']/1e3) + 5
    ax.scatter(
        group_df_dr2['phot_bp_mean_mag']-group_df_dr2['phot_rp_mean_mag'],
        yval,
        c='k', alpha=1., zorder=3, s=5, rasterized=True, linewidths=0,
        label='Members'# 'CG18 P>0.1'
    )

    target_yval = np.array([target_df['phot_g_mean_mag'] +
                            5*np.log10(target_df['parallax']/1e3) + 5])
    ax.plot(
        target_df['phot_bp_mean_mag']-target_df['phot_rp_mean_mag'],
        target_yval,
        alpha=1, mew=0.5, zorder=8, label='TOI 837', markerfacecolor='yellow',
        markersize=9, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    ax.set_ylabel('G + $5\log_{10}(\omega_{\mathrm{as}}) + 5$', fontsize='large')
    ax.set_xlabel('Bp - Rp', fontsize='large')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    format_ax(ax)
    outpath = os.path.join(outdir, 'hr.png')
    savefig(f, outpath)


def plot_positions(outdir):

    # from cdips.tests.test_nbhd_plot
    pklpath = '/Users/luke/Dropbox/proj/timmy/results/cluster_membership/nbhd_info_5251470948229949568.pkl'
    info = pickle.load(open(pklpath, 'rb'))
    (targetname, groupname, group_df_dr2, target_df, nbhd_df,
     cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
     group_in_k13, group_in_cg18, group_in_kc19, group_in_k18
    ) = info

    ##########

    plt.close('all')

    # ra vs ra 
    # dec vs ra  ---  dec vs dec       
    # parallax vs ra  ---  parallax vs dec --- parallax vs parallax

    f, axs = plt.subplots(figsize=(4,4), nrows=2, ncols=2)

    ax_ixs = [(0,0),(1,0),(1,1)]
    xy_tups = [('ra', 'dec'), ('ra', 'parallax'), ('dec', 'parallax')]
    ldict = {
        'ra': r'$\alpha$ [deg]',
        'dec': r'$\delta$ [deg]',
        'parallax': r'$\pi$ [mas]'
    }

    for ax_ix, xy_tup in zip(ax_ixs, xy_tups):

        i, j = ax_ix[0], ax_ix[1]
        xv, yv = xy_tup[0], xy_tup[1]

        axs[i,j].scatter(
            nbhd_df[xv], nbhd_df[yv], c='gray', alpha=0.9, zorder=2, s=5,
            rasterized=True, linewidths=0, label='Neighborhood', marker='.'
        )
        axs[i,j].scatter(
            group_df_dr2[xv], group_df_dr2[yv], c='k', alpha=0.9,
            zorder=3, s=5, rasterized=True, linewidths=0, label='Members'
        )
        axs[i,j].plot(
            target_df[xv], target_df[yv], alpha=1, mew=0.5, zorder=8,
            label='TOI 837', markerfacecolor='yellow', markersize=9, marker='*',
            color='black', lw=0
        )

    axs[0,0].set_ylabel(ldict['dec'])

    axs[1,0].set_xlabel(ldict['ra'])
    axs[1,0].set_ylabel(ldict['parallax'])

    axs[1,1].set_xlabel(ldict['dec'])

    axs[0,1].set_axis_off()

    # ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)

    for ax in axs.flatten():
        format_ax(ax)

    outpath = os.path.join(outdir, 'positions.png')
    savefig(f, outpath)


def plot_velocities(outdir):

    # from cdips.tests.test_nbhd_plot
    pklpath = '/Users/luke/Dropbox/proj/timmy/results/cluster_membership/nbhd_info_5251470948229949568.pkl'
    info = pickle.load(open(pklpath, 'rb'))
    (targetname, groupname, group_df_dr2, target_df, nbhd_df,
     cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
     group_in_k13, group_in_cg18, group_in_kc19, group_in_k18
    ) = info

    ##########

    plt.close('all')

    f, axs = plt.subplots(figsize=(4,4), nrows=2, ncols=2)

    ax_ixs = [(0,0),(1,0),(1,1)]
    xy_tups = [('pmra', 'pmdec'),
               ('pmra', 'radial_velocity'),
               ('pmdec', 'radial_velocity')]
    ldict = {
        'pmra': r'$\mu_{{\alpha}} \cos\delta$ [mas/yr]',
        'pmdec':  r'$\mu_{{\delta}}$ [mas/yr]',
        'radial_velocity': 'RV [km/s]'
    }

    for ax_ix, xy_tup in zip(ax_ixs, xy_tups):

        i, j = ax_ix[0], ax_ix[1]
        xv, yv = xy_tup[0], xy_tup[1]

        axs[i,j].scatter(
            nbhd_df[xv], nbhd_df[yv], c='gray', alpha=0.9, zorder=2, s=5,
            rasterized=True, linewidths=0, label='Neighborhood', marker='.'
        )
        axs[i,j].scatter(
            group_df_dr2[xv], group_df_dr2[yv], c='k', alpha=0.9,
            zorder=3, s=5, rasterized=True, linewidths=0, label='Members'
        )
        axs[i,j].plot(
            target_df[xv], target_df[yv], alpha=1, mew=0.5, zorder=8,
            label='TOI 837', markerfacecolor='yellow', markersize=9, marker='*',
            color='black', lw=0
        )

    axs[0,0].set_ylabel(ldict['pmdec'])

    axs[1,0].set_xlabel(ldict['pmra'])
    axs[1,0].set_ylabel(ldict['radial_velocity'])

    axs[1,1].set_xlabel(ldict['pmdec'])

    axs[0,1].set_axis_off()

    # ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)

    for ax in axs.flatten():
        format_ax(ax)

    outpath = os.path.join(outdir, 'velocities.png')
    savefig(f, outpath)


def plot_full_kinematics(outdir):

    # from cdips.tests.test_nbhd_plot
    pklpath = '/Users/luke/Dropbox/proj/timmy/results/cluster_membership/nbhd_info_5251470948229949568.pkl'
    info = pickle.load(open(pklpath, 'rb'))
    (targetname, groupname, group_df_dr2, target_df, nbhd_df,
     cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
     group_in_k13, group_in_cg18, group_in_kc19, group_in_k18
    ) = info

    ##########

    plt.close('all')

    params = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
    nparams = len(params)

    qlimd = {
        'ra': 0,
        'dec': 0,
        'parallax': 0,
        'pmra': 1,
        'pmdec': 1,
        'radial_velocity': 1
    } # whether to limit axis by quantile

    ldict = {
        'ra': r'$\alpha$ [deg]',
        'dec': r'$\delta$ [deg]',
        'parallax': r'$\pi$ [mas]',
        'pmra': r'$\mu_{{\alpha}} \cos\delta$ [mas/yr]',
        'pmdec':  r'$\mu_{{\delta}}$ [mas/yr]',
        'radial_velocity': 'RV [km/s]'
    }

    f, axs = plt.subplots(figsize=(8,8), nrows=nparams-1, ncols=nparams-1)

    for i in range(nparams):
        for j in range(nparams):
            print(i,j)
            if j == nparams-1 or i == nparams-1:
                continue
            if j>i:
                axs[i,j].set_axis_off()
                continue

            xv = params[j]
            yv = params[i+1]
            print(i,j,xv,yv)

            axs[i,j].scatter(
                nbhd_df[xv], nbhd_df[yv], c='gray', alpha=0.9, zorder=2, s=5,
                rasterized=True, linewidths=0, label='Neighborhood', marker='.'
            )
            axs[i,j].scatter(
                group_df_dr2[xv], group_df_dr2[yv], c='k', alpha=0.9,
                zorder=3, s=5, rasterized=True, linewidths=0, label='Members'
            )
            axs[i,j].plot(
                target_df[xv], target_df[yv], alpha=1, mew=0.5, zorder=8,
                label='TOI 837', markerfacecolor='yellow', markersize=9, marker='*',
                color='black', lw=0
            )

            # set the axis limits as needed
            if qlimd[xv]:
                xlim = (np.nanpercentile(nbhd_df[xv], 25),
                        np.nanpercentile(nbhd_df[xv], 75))
                axs[i,j].set_xlim(xlim)
            if qlimd[yv]:
                ylim = (np.nanpercentile(nbhd_df[yv], 25),
                        np.nanpercentile(nbhd_df[yv], 75))
                axs[i,j].set_ylim(ylim)

            # fix labels
            if j == 0 :
                axs[i,j].set_ylabel(ldict[yv])
                if not i == nparams - 2:
                    # hide xtick labels
                    labels = [item.get_text() for item in axs[i,j].get_xticklabels()]
                    empty_string_labels = ['']*len(labels)
                    axs[i,j].set_xticklabels(empty_string_labels)

            if i == nparams - 2:
                axs[i,j].set_xlabel(ldict[xv])
                if not j == 0:
                    # hide ytick labels
                    labels = [item.get_text() for item in axs[i,j].get_yticklabels()]
                    empty_string_labels = ['']*len(labels)
                    axs[i,j].set_yticklabels(empty_string_labels)

            if (not (j == 0)) and (not (i == nparams - 2)):
                # hide ytick labels
                labels = [item.get_text() for item in axs[i,j].get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                axs[i,j].set_yticklabels(empty_string_labels)
                # hide xtick labels
                labels = [item.get_text() for item in axs[i,j].get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                axs[i,j].set_xticklabels(empty_string_labels)

    # axs[2,2].legend(loc='best', handletextpad=0.1, fontsize='medium', framealpha=0.7)
    # leg = axs[2,2].legend(bbox_to_anchor=(0.8,0.8), loc="upper right",
    #                       handletextpad=0.1, fontsize='medium',
    #                       bbox_transform=f.transFigure)


    for ax in axs.flatten():
        format_ax(ax)

    f.tight_layout(h_pad=0.1, w_pad=0.1)

    outpath = os.path.join(outdir, 'full_kinematics.png')
    savefig(f, outpath)


def plot_groundscene(c_obj, img_wcs, img, outpath, Tmag_cutoff=17,
                     showcolorbar=0, ticid=None, xlim=None, ylim=None,
                     ap_mask=0, customap=0):

    plt.close('all')

    # standard tick formatting fails for these images.
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    #
    # wcs information parsing
    # follow Clara Brasseur's https://github.com/ceb8/tessworkshop_wcs_hack
    # (this is from the CDIPS vetting reports...)
    #
    radius = 6.0*u.arcminute

    nbhr_stars = Catalogs.query_region(
        "{} {}".format(float(c_obj.ra.value), float(c_obj.dec.value)),
        catalog="TIC",
        radius=radius
    )

    try:
        px,py = img_wcs.all_world2pix(
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ra'],
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['dec'],
            0
        )
    except Exception as e:
        print('ERR! wcs all_world2pix got {}'.format(repr(e)))
        raise(e)

    ticids = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ID']
    tmags = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['Tmag']

    sel = (px > 0) & (px < img.shape[1]) & (py > 0) & (py < img.shape[0])
    if isinstance(ticid, str):
        sel &= (ticids != ticid)

    px,py = px[sel], py[sel]
    ticids, tmags = ticids[sel], tmags[sel]

    ra, dec = float(c_obj.ra.value), float(c_obj.dec.value)
    target_x, target_y = img_wcs.all_world2pix(ra,dec,0)

    # geometry: there are TWO coordinate axes. (x,y) and (ra,dec). To get their
    # relative orientations, the WCS and ignoring curvature will usually work.
    shiftra_x, shiftra_y = img_wcs.all_world2pix(ra+1e-4,dec,0)
    shiftdec_x, shiftdec_y = img_wcs.all_world2pix(ra,dec+1e-4,0)

    ##########################################

    plt.close('all')
    fig = plt.figure(figsize=(5,5))

    # ax: whatever the groundbased image was
    ax = plt.subplot2grid((1, 1), (0, 0), projection=img_wcs)

    ##########################################

    #
    # ax0: img
    #

    #interval = vis.PercentileInterval(99.99)
    #interval = vis.AsymmetricPercentileInterval(1,99.9)
    vmin,vmax = 10, int(1e4)
    norm = vis.ImageNormalize(
        vmin=vmin, vmax=vmax, stretch=vis.LogStretch(1000))

    cset0 = ax.imshow(img, cmap=plt.cm.gray, origin='lower', zorder=1,
                      norm=norm)

    if isinstance(ap_mask, np.ndarray):
        for x,y in product(range(10),range(10)):
            if ap_mask[y,x]:
                ax.add_patch(
                    patches.Rectangle(
                        (x-.5, y-.5), 1, 1, hatch='//', fill=False, snap=False,
                        linewidth=0., zorder=2, alpha=0.7, rasterized=True
                    )
                )

    ax.scatter(px, py, marker='o', c='C1', s=2e4/(tmags**3), rasterized=True,
                zorder=6, linewidths=0.8)
    # ax0.scatter(px, py, marker='x', c='C1', s=20, rasterized=True,
    #             zorder=6, linewidths=0.8)
    ax.plot(target_x, target_y, mew=0.5, zorder=5, markerfacecolor='yellow',
            markersize=10, marker='*', color='k', lw=0)

    if customap:
        datestr = [e for e in outpath.split('/') if '2020' in e][0]
        tdir = (
            '/Users/luke/Dropbox/proj/timmy/results/groundphot/{}/photutils_apphot/'.
            format(datestr)
        )
        inpath = os.path.join(
            tdir,
            os.path.basename(outpath).replace(
                'groundscene.png', 'customtable.fits')
        )
        chdul = fits.open(inpath)
        d = chdul[1].data
        chdul.close()
        xc, yc = d['xcenter'], d['ycenter']
        colors = ['C{}'.format(ix) for ix in range(len(xc))]
        for _x, _y, _c in zip(xc, yc, colors):
            ax.scatter(_x, _y, marker='x', c=_c, s=20, rasterized=True,
                       zorder=6, linewidths=0.8)

    ax.set_title('El Sauce 36cm', fontsize='xx-large')

    if showcolorbar:
        cb0 = fig.colorbar(cset0, ax=ax, extend='neither', fraction=0.046, pad=0.04)

    #
    # fix the axes
    #
    ax.grid(ls='--', alpha=0.5)
    if shiftra_x - target_x > 0:
        # want RA to increase to the left (almost E)
        ax.invert_xaxis()
    if shiftdec_y - target_y < 0:
        # want DEC to increase up (almost N)
        ax.invert_yaxis()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    format_ax(ax)
    ax.set_xlabel(r'$\alpha_{2000}$')
    ax.set_ylabel(r'$\delta_{2000}$')

    if showcolorbar:
        fig.tight_layout(h_pad=-8, w_pad=-8)
    else:
        fig.tight_layout(h_pad=1, w_pad=1)

    savefig(fig, outpath, writepdf=0, dpi=300)


def shift_img_plot(img, shift_img, xlim, ylim, outpath, x0, y0, target_x,
                   target_y, titlestr0, titlestr1, showcolorbar=1):

    vmin,vmax = 10, int(1e4)
    norm = vis.ImageNormalize(
        vmin=vmin, vmax=vmax, stretch=vis.LogStretch(1000))

    fig, axs = plt.subplots(nrows=2,ncols=1)
    axs[0].imshow(img, cmap=plt.cm.gray, origin='lower', norm=norm)
    axs[0].set_title(titlestr0)
    axs[0].plot(target_x, target_y, mew=0.5, zorder=5,
                markerfacecolor='yellow', markersize=3, marker='*', color='k',
                lw=0)

    cset = axs[1].imshow(shift_img, cmap=plt.cm.gray, origin='lower', norm=norm)
    axs[1].set_title(titlestr1)
    axs[1].plot(x0, y0, mew=0.5, zorder=5, markerfacecolor='yellow',
                markersize=3, marker='*', color='k', lw=0)

    for ax in axs:
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        format_ax(ax)

    if showcolorbar:
        raise NotImplementedError
        cb0 = fig.colorbar(cset, ax=axs[1], extend='neither', fraction=0.046, pad=0.04)

    fig.tight_layout()

    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_pixel_lc(times, img_cube, outpath, showvlines=0):
    # 20x20 around target pixel
    nrows, ncols = 20, 20
    fig, axs = plt.subplots(figsize=(20,20), nrows=nrows, ncols=ncols,
                            sharex=True)

    x0, y0 = 768, 512  # in array coordinates
    xmin, xmax = x0-10, x0+10  # note: xmin/xmax in mpl coordinates (not ndarray coordinates)
    ymin, ymax = y0-10, y0+10 # note: ymin/ymax in mpl coordinates (not ndarray coordinates)

    props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
                 linewidth=0)

    time_offset = np.nanmin(times)
    times -= time_offset

    N_drop = 47 # per Phil Evan's reduction notes

    for ax_i, data_i in enumerate(range(xmin, xmax)):
        for ax_j, data_j in enumerate(list(range(ymin, ymax))[::-1]):
           # note: y reverse is the same as "origin = lower"

            print(ax_i, ax_j, data_i, data_j)

            axs[ax_j,ax_i].scatter(
                times[N_drop:], img_cube[N_drop:, data_j, data_i], c='k', zorder=3, s=2,
                rasterized=True, linewidths=0
            )

            tstr = (
                '{:.1f}\n{} {}'.format(
                    np.nanpercentile( img_cube[N_drop:, data_j, data_i], 99),
                    data_i, data_j)
            )
            axs[ax_j,ax_i].text(0.97, 0.03, tstr, ha='right', va='bottom',
                                transform=axs[ax_j,ax_i].transAxes, bbox=props,
                                zorder=-1, fontsize='xx-small')

            if showvlines:
                ylim = axs[ax_j,ax_i].get_ylim()
                axs[ax_j,ax_i].vlines(
                    [2458940.537 - time_offset, 2458940.617 - time_offset],
                    min(ylim), max(ylim), colors='C1', alpha=0.5, linestyles='--',
                    zorder=-2, linewidths=1
                )
                axs[ax_j,ax_i].set_ylim(ylim)

            # hide ytick labels
            labels = [item.get_text() for item in axs[ax_j,ax_i].get_yticklabels()]
            empty_string_labels = ['']*len(labels)
            axs[ax_j,ax_i].set_yticklabels(empty_string_labels)

            format_ax(axs[ax_j,ax_i])

    fig.tight_layout(h_pad=0, w_pad=0)

    savefig(fig, outpath, writepdf=0, dpi=300)


def vis_photutils_lcs(datestr, ap, overwrite=1):

    outpath = os.path.join(
        RESULTSDIR,  'groundphot', datestr, 'vis_photutils_lcs',
        'vis_photutils_lcs_{}.png'.format(ap)
    )

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    lcdir = '/Users/luke/Dropbox/proj/timmy/results/groundphot/{}/vis_photutils_lcs'.format(datestr)
    lcpaths = glob(os.path.join(lcdir, 'TIC*csv'))

    lcs = [pd.read_csv(l) for l in lcpaths]

    target_ticid = '460205581' # TOI 837
    target_lc = pd.read_csv(glob(os.path.join(
        lcdir, 'TIC*{}*csv'.format(target_ticid)))[0]
    )

    N_trim = 47 # drop the first 47 points due to clouds

    time = target_lc['BJD_TDB'][N_trim:]
    flux = target_lc[ap][N_trim:]
    mean_flux = np.nanmean(flux)
    flux /= mean_flux

    print(42*'-')
    print(target_ticid, target_lc.id.iloc[0], mean_flux)
    #FIXME : something is wrong with the way you are doing this. you are not
    # not correctly selecting bright stars (!!!!) check the logic in the
    # following "comp_ind" selection

    comp_mean_fluxs = nparr([np.nanmean(lc[ap][N_trim:]) for lc in lcs])
    comp_inds = np.argsort(np.abs(mean_flux - comp_mean_fluxs))

    # take the top like ... 9 say. turns out, TOI837 is the brightest of them!
    N_comp = 9

    #
    # finally, make the plot
    #
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(time, flux, c='k', zorder=3, s=3, rasterized=True, linewidths=0)

    tstr = 'TIC{}'.format(target_ticid)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5, pad=0.15,
                 linewidth=0)
    ax.text(np.nanpercentile(time, 97), np.nanpercentile(flux, 3), tstr,
            ha='right', va='top', bbox=props, zorder=-1, fontsize='small')

    offset = 0.3

    outdf = pd.DataFrame({})
    for ix, comp_ind in enumerate(comp_inds[1:N_comp+1]):

        lc = lcs[comp_ind]

        time = lc['BJD_TDB'][N_trim:]
        flux = lc[ap][N_trim:]
        mean_flux = np.nanmean(flux)
        flux /= mean_flux

        print(lc['ticid'].iloc[0], lc.id.iloc[0], mean_flux)

        color = 'C{}'.format(ix)
        ax.scatter(time, flux+offset, s=3, rasterized=True, linewidths=0,
                   c=color)

        tstr = 'TIC{}'.format(lc['ticid'].iloc[0])
        ax.text(np.nanpercentile(time, 97), np.nanpercentile(flux+offset, 50),
                tstr, ha='right', va='top', bbox=props, zorder=-1,
                fontsize='small', color=color)

        offset += 0.3

        t = lc['ticid'].iloc[0]
        outdf['time_{}'.format(t)] = time
        outdf['flux_{}'.format(t)] = flux

    outcsvpath = os.path.join(
        RESULTSDIR,  'groundphot', datestr, 'vis_photutils_lcs',
        'vis_photutils_lcs_compstars_{}.csv'.format(ap)
    )

    outdf.to_csv(outcsvpath, index=False)
    print('made {}'.format(outcsvpath))

    format_ax(ax)

    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)


def stackviz_blend_check(datestr, apn, soln=0, overwrite=1):

    if soln == 1:
        raise NotImplementedError('gotta implement image cheat')

    outdir = os.path.join(RESULTSDIR,  'groundphot', datestr,
                          'stackviz_blend_check')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'stackviz_blend_check_{}.png'.format(apn))

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    lcdir = '/Users/luke/Dropbox/proj/timmy/results/groundphot/{}/compstar_detrend'.format(datestr)
    lcpaths = np.sort(glob(os.path.join(
        lcdir, 'toi837_detrended*_sum_{}_*.csv'.format(apn))))
    assert len(lcpaths) == 13

    origlcdir = '/Users/luke/Dropbox/proj/timmy/results/groundphot/{}/vis_photutils_lcs'.format(datestr)

    lcs = [pd.read_csv(l) for l in lcpaths]

    N_lcs = len(lcpaths)
    #
    # make the plot
    #
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8,N_lcs))

    offset = 0
    props = dict(boxstyle='square', facecolor='white', alpha=0.5, pad=0.15,
                 linewidth=0)

    for ix, lc in enumerate(lcs):

        time = nparr(lc['time'])
        flux = nparr(lc['flat_flux'])

        _id = str(ix+1).zfill(4)
        origpath = os.path.join(
            origlcdir, 'CUSTOM{}_photutils_groundlc.csv'.format(_id)
        )
        odf = pd.read_csv(origpath)
        ra, dec = np.mean(odf['sky_center.ra']), np.mean(odf['sky_center.dec'])

        print(_id, ra, dec)

        color = 'C{}'.format(ix)
        ax.scatter(time, flux+offset, s=3, rasterized=True, linewidths=0,
                   c=color)

        tstr = '{}: {:.4f} {:.4f}'.format(_id, ra, dec)
        ax.text(np.nanpercentile(time, 97), np.nanpercentile(flux+offset, 1),
                tstr, ha='right', va='bottom', bbox=props, zorder=-1,
                fontsize='small', color=color)

        if int(apn) <= 1:
            offset += 0.30
        elif int(apn) == 2:
            offset += 0.10
        elif int(apn) == 3:
            offset += 0.05
        elif int(apn) == 4:
            offset += 0.035
        else:
            offset += 0.02

    format_ax(ax)

    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)
