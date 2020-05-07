"""
Plots:

    plot_quicklooklc
    plot_raw_zoom

    plot_MAP_data
    plot_sampleplot
    plot_phasefold_map
    plot_phasefold_post
    plot_traceplot
    plot_cornerplot

    plot_scene

    plot_hr
    plot_astrometric_excess
"""
import os, corner, pickle
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from numpy import array as nparr
from itertools import product

from billy.plotting import savefig, format_ax
import billy.plotting as bp
from billy.plotting import plot_phasefold_map, plot_traceplot, plot_cornerplot

from timmy.paths import DATADIR, RESULTSDIR
from timmy.convenience import get_data, get_clean_data, detrend_data

from astrobase.lcmath import (
    phase_magseries, phase_bin_magseries, sigclip_magseries,
    find_lc_timegroups
)
from astrobase import periodbase

from astropy.stats import LombScargle
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.time import Time


##################################################
# wrappers to generic plots implemented in billy #
##################################################

def plot_test_data(x_obs, y_obs, y_mod, modelid, outdir):
    bp.plot_test_data(x_obs, y_obs, y_mod, modelid, outdir)


def plot_MAP_data(x_obs, y_obs, y_MAP, outpath):
    bp.plot_MAP_data(x_obs, y_obs, y_MAP, outpath, ms=1)


def plot_sampleplot(m, outpath, N_samples=100):
    bp.plot_sampleplot(m, outpath, N_samples=N_samples)


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

    t0 = 1574.2738
    per = 8.32467
    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    plt.close('all')

    f,axs = plt.subplots(figsize=(25,8), nrows=2, sharex=True)

    xmin, xmax = np.nanmin(time)-1, np.nanmax(time)+1

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

    for ax in axs:

        # ymin, ymax = ax.get_ylim()

        # ax.vlines(
        #     tra_times, ymin, ymax, colors='C1', alpha=0.5,
        #     linestyles='--', zorder=-2, linewidths=0.5
        # )

        # ax.set_ylim((ymin, ymax))

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

        ymin, ymax = ax.get_ylim()
        ax.vlines(
            mid_time, ymin, ymax, colors='C1', alpha=0.5,
            linestyles='--', zorder=-2, linewidths=0.5
        )
        ax.set_ylim((ymin, ymax))


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

    fig.tight_layout(h_pad=0.2, w_pad=-0.5)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_phasefold_map(m, d, outpath):

    if os.path.exists(outpath) and not m.OVERWRITE:
        return

    # recover periods and epochs.
    P_rot = 2*np.pi/float(m.map_estimate['omegarot'])
    t0_rot = float(m.map_estimate['phirot']) * P_rot / (2*np.pi)
    P_orb = float(m.map_estimate['period'])
    t0_orb = float(m.map_estimate['t0'])

    # phase and bin them.
    orb_d = phase_magseries(
        d['x_obs'], d['y_orb'], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=0.01
    )
    rot_d = phase_magseries(
        d['x_obs'], d['y_rot'], P_rot, t0_rot, wrap=True, sort=True
    )
    rot_bd = phase_bin_magseries(
        rot_d['phase'], rot_d['mags'], binsize=0.01
    )

    # make tha plot
    plt.close('all')
    fig, axs = plt.subplots(nrows=2, figsize=(6, 6), sharex=True)

    axs[0].scatter(rot_d['phase'], rot_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0, rasterized=True)
    axs[0].scatter(rot_bd['binnedphases'], rot_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)

    txt0 = '$P_{{\mathrm{{\ell}}}}$ = {:.5f}$\,$d'.format(P_rot)
    props = dict(boxstyle='square', facecolor='white', alpha=0.9, pad=0.15,
                 linewidth=0)

    axs[0].text(0.98, 0.98, txt0, ha='right', va='top',
                transform=axs[0].transAxes, bbox=props, zorder=3)
    #axs[0].set_ylabel('$f_{{\mathrm{{\ell}}}} = f - f_{{\mathrm{{s}}}}$',
    #                  fontsize='large')
    axs[0].set_ylabel('Longer period', fontsize='large')

    axs[1].scatter(orb_d['phase'], orb_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0, rasterized=True)
    axs[1].scatter(orb_bd['binnedphases'], orb_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)

    out_d = {
        'orb_d': orb_d,
        'orb_bd': orb_bd,
        'rot_d': rot_d,
        'rot_bd': rot_bd
    }
    pklpath = os.path.join(
        os.path.dirname(outpath),
        os.path.basename(outpath).replace('.png','_points.pkl')
    )
    with open(pklpath, 'wb') as buff:
        pickle.dump(out_d, buff)
    print('made {}'.format(pklpath))

    txt1 = '$P_{{\mathrm{{s}}}}$ = {:.5f}$\,$d'.format(P_orb)
    axs[1].text(0.98, 0.98, txt1, ha='right', va='top',
                transform=axs[1].transAxes, bbox=props, zorder=3)

    axs[1].set_ylabel('Shorter period',
                      fontsize='large')
    #axs[1].set_ylabel('$f_{{\mathrm{{s}}}} = f - f_{{\mathrm{{\ell}}}}$',
    #                  fontsize='large')

    axs[1].set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    axs[1].set_yticks([-0.04, -0.02, 0, 0.02, 0.04])

    axs[-1].set_xlabel('Phase', fontsize='large')

    for a in axs:
        a.grid(which='major', axis='both', linestyle='--', zorder=-3,
               alpha=0.5, color='gray', linewidth=0.5)

    # pct_80 = np.percentile(results.model_folded_model, 80)
    # pct_20 = np.percentile(results.model_folded_model, 20)
    # center = np.nanmedian(results.model_folded_model)
    # delta_y = (10/6)*np.abs(pct_80 - pct_20)
    # plt.ylim(( center-0.7*delta_y, center+0.7*delta_y ))

    for a in axs:
        a.set_xlim((-1, 1))
        format_ax(a)
    axs[0].set_ylim((-0.075, 0.075))
    axs[1].set_ylim((-0.045, 0.045))
    fig.tight_layout()
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_phasefold_post(m, d, outpath):

    # recover periods and epochs.
    P_rot = 2*np.pi/float(np.nanmedian(m.trace['omegarot']))
    t0_rot = float(np.nanmedian(m.trace['phirot'])) * P_rot / (2*np.pi)
    P_orb = float(np.nanmedian(m.trace['period']))
    t0_orb = float(np.nanmedian(m.trace['t0']))

    # phase and bin them.
    orb_d = phase_magseries(
        d['x_obs'], d['y_orb'], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=0.01
    )
    rot_d = phase_magseries(
        d['x_obs'], d['y_rot'], P_rot, t0_rot, wrap=True, sort=True
    )
    rot_bd = phase_bin_magseries(
        rot_d['phase'], rot_d['mags'], binsize=0.01
    )

    # make tha plot
    plt.close('all')
    fig, axs = plt.subplots(nrows=2, figsize=(6, 8), sharex=True)

    axs[0].scatter(rot_d['phase'], rot_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0)
    axs[0].scatter(rot_bd['binnedphases'], rot_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)
    txt0 = 'Prot {:.5f}d, t0 {:.5f}'.format(P_rot, t0_rot)
    axs[0].text(0.98, 0.98, txt0, ha='right', va='top',
                transform=axs[0].transAxes)
    axs[0].set_ylabel('flux-orb (rot)')

    axs[1].scatter(orb_d['phase'], orb_d['mags'], color='gray', s=2, alpha=0.8,
                   zorder=4, linewidths=0)
    axs[1].scatter(orb_bd['binnedphases'], orb_bd['binnedmags'], color='black',
                   s=8, alpha=1, zorder=5, linewidths=0)
    txt1 = 'Porb {:.5f}d, t0 {:.5f}'.format(P_orb, t0_orb)
    axs[1].text(0.98, 0.98, txt1, ha='right', va='top',
                transform=axs[1].transAxes)
    axs[1].set_ylabel('flux-rot (orb)')
    axs[1].set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

    axs[-1].set_xlabel('phase')

    for a in axs:
        a.grid(which='major', axis='both', linestyle='--', zorder=-3,
                 alpha=0.5, color='gray')

    # pct_80 = np.percentile(results.model_folded_model, 80)
    # pct_20 = np.percentile(results.model_folded_model, 20)
    # center = np.nanmedian(results.model_folded_model)
    # delta_y = (10/6)*np.abs(pct_80 - pct_20)
    # plt.ylim(( center-0.7*delta_y, center+0.7*delta_y ))

    for a in axs:
        a.set_xlim((-0.1-0.5, 1.1-0.5))
        format_ax(a)
    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)



def plot_scene(c_obj, img_wcs, img, outpath, Tmag_cutoff=17, showcolorbar=0,
               ap_mask=0, bkgd_mask=0, ticid=None):

    from astrobase.plotbase import skyview_stamp
    from astropy.wcs import WCS
    from astroquery.mast import Catalogs
    import astropy.visualization as vis
    import matplotlib as mpl
    from matplotlib import patches

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

    #FIXME
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

    pklpath = '/Users/luke/Dropbox/proj/billy/results/cluster_membership/nbhd_info_3222255959210123904.pkl'
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
        c='gray', alpha=1., zorder=2, s=7, rasterized=True, linewidths=0,
        label='Neighborhood', marker='.'
    )

    yval = group_df_dr2['phot_g_mean_mag'] + 5*np.log10(group_df_dr2['parallax']/1e3) + 5
    ax.scatter(
        group_df_dr2['phot_bp_mean_mag']-group_df_dr2['phot_rp_mean_mag'],
        yval,
        c='k', alpha=1., zorder=3, s=9, rasterized=True, linewidths=0,
        label='K+18 members'
    )

    target_yval = np.array([target_df['phot_g_mean_mag'] +
                            5*np.log10(target_df['parallax']/1e3) + 5])
    ax.plot(
        target_df['phot_bp_mean_mag']-target_df['phot_rp_mean_mag'],
        target_yval,
        alpha=1, mew=0.5, zorder=8, label='PTFO 8-8695', markerfacecolor='yellow',
        markersize=12, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', handletextpad=0.1, fontsize='small')
    ax.set_ylabel('G + $5\log_{10}(\omega_{\mathrm{as}}) + 5$', fontsize='large')
    ax.set_xlabel('Bp - Rp', fontsize='large')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    ax.set_xlim((0.9, 3.1))
    ax.set_ylim((9.5, 4.5))

    # # set M_omega y limits
    # min_y = np.nanmin(np.array([np.nanpercentile(nbhd_yval, 2), target_yval]))
    # max_y = np.nanmax(np.array([np.nanpercentile(nbhd_yval, 98), target_yval]))
    # edge_y = 0.01*(max_y - min_y)
    # momega_ylim = [max_y+edge_y, min_y-edge_y]
    # ax.set_ylim(momega_ylim)

    format_ax(ax)
    outpath = os.path.join(outdir, 'hr.png')
    savefig(f, outpath)
