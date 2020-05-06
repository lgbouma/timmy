"""
Plots:

    plot_quicklooklc

    plot_periodogram
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
from datetime import datetime
from pymc3.backends.tracetab import trace_to_dataframe
from itertools import product

from billy.plotting import savefig, format_ax

from billy.plotting import (
    plot_periodogram, plot_MAP_data, plot_sampleplot, plot_phasefold_map,
    plot_traceplot, plot_cornerplot
)

from timmy.paths import DATADIR, RESULTSDIR

#FIXME
#FIXME cleanup
#FIXME
from billy.convenience import flatten as bflatten
from billy.convenience import get_clean_ptfo_data
from billy.models import linear_model

from astrobase.lcmath import (
    phase_magseries, phase_bin_magseries, sigclip_magseries,
    find_lc_timegroups
)
from astrobase import periodbase

from astropy.stats import LombScargle
from astropy import units as u, constants as const
from astropy.io import fits
from astropy.time import Time

from numpy import array as nparr

from timmy.convenience import get_data, get_clean_data

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

    f,axs = plt.subplots(figsize=(25,8), nrows=2)

    xmin, xmax = np.nanmin(time)-1, np.nanmax(time)+1

    s = 2.5 if provenance == 'spoc' else 2.5*10
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

        ymin, ymax = ax.get_ylim()

        ax.vlines(
            tra_times, ymin, ymax, colors='C1', alpha=0.5,
            linestyles='--', zorder=-2, linewidths=0.5
        )

        ax.set_ylim((ymin, ymax))
        ax.set_xlim((xmin, xmax))

        format_ax(ax)

    f.tight_layout(h_pad=0., w_pad=0.)
    savefig(f, outpath, writepdf=0, dpi=300)





def plot_periodogram(outdir, islinear=True):

    x_obs, y_obs, y_err = get_clean_ptfo_data(binsize=None)

    period_min, period_max, N_freqs = 0.3, 0.7, int(3e3)
    frequency = np.linspace(1/period_max, 1/period_min, N_freqs)
    ls = LombScargle(x_obs, y_obs, y_err, normalization='standard')
    power = ls.power(frequency)
    period = 1/frequency

    P_rot, P_orb = 0.49914, 0.4485

    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3))
    ax.plot(
        period, power, lw=0.5, c='k'
    )

    if not islinear:
        ax.set_yscale('log')

    ylim = ax.get_ylim()
    for P,c in zip([P_rot, P_orb],['C0','C1']):
        for m in [1]:
            ax.vlines(
                m*P, min(ylim), max(ylim), colors=c, alpha=0.5,
                linestyles='--', zorder=-2, linewidths=0.5
            )

    #ax.set_xlabel('Frequency [1/days]')
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('Lomb-Scargle Power')
    if not islinear:
        ax.set_ylim([1e-4, 1.2])
    ax.set_xlim([period_min, period_max])

    format_ax(ax)
    outpath = os.path.join(outdir, 'periodogram.png')
    savefig(f, outpath)


def plot_test_data(x_obs, y_obs, y_mod, modelid, outdir):
    plt.close('all')
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111, xlabel='x_obs', ylabel='y_obs',
                         title='Generated data and underlying model')
    ax.plot(x_obs, y_obs, 'x', label='sampled data')
    ax.plot(x_obs, y_mod, label='true regression line', lw=2.)
    plt.legend(loc=0)
    outpath = os.path.join(outdir, 'test_{}_data.png'.format(modelid))
    format_ax(ax)
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_MAP_data(x_obs, y_obs, y_MAP, outpath):
    plt.close('all')
    plt.figure(figsize=(14, 4))
    plt.plot(x_obs, y_obs, ".k", ms=4, label="data")
    plt.plot(x_obs, y_MAP, lw=1)
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    _ = plt.title("MAP model")
    fig = plt.gcf()
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_sampleplot(m, outpath, N_samples=100):

    if os.path.exists(outpath) and not m.OVERWRITE:
        return

    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(m.x_obs, m.y_obs, ".k", ms=4, label="data", zorder=N_samples+1)
    ax.plot(m.x_obs, m.map_estimate['mu_model'], lw=0.5, label='MAP',
            zorder=N_samples+2, color='C1', alpha=1)

    np.random.seed(42)
    y_mod_samples = (
        m.trace.mu_model[
            np.random.choice(
                m.trace.mu_model.shape[0], N_samples, replace=False
            ), :
        ]
    )

    for i in range(N_samples):
        if i % 10 == 0:
            print(i)
        ax.plot(m.x_obs, y_mod_samples[i,:], color='C0', alpha=0.3,
                rasterized=True, lw=0.5)

    ax.set_ylabel("relative flux")
    ax.set_xlabel("time [days]")
    ax.legend(loc='best')
    savefig(fig, outpath, writepdf=0, dpi=300)


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


def plot_splitsignal_post(m, outpath):
    """
    y_obs + y_mod + y_rot + y_orb
    things at rotation frequency
    things at orbital frequency
    """

    # get y_mod, y_rot, y_orb, y_tra. here: cheat. just randomly select 1 from
    # posterior (TODO: take the median parameters, +generate the model instead)
    np.random.seed(42)
    sel = np.random.choice(m.trace.mu_model.shape[0], 1)
    y_mod = m.trace.mu_model[sel, :].flatten()
    y_tra = m.trace.mu_transit[sel, :].flatten()

    y_orb, y_rot = np.zeros_like(m.x_obs), np.zeros_like(m.x_obs)
    for modelcomponent in m.modelcomponents:
        if 'rot' in modelcomponent:
            N_harmonics = int(modelcomponent[0])
            for ix in range(N_harmonics):
                y_rot += m.trace['mu_rotsin{}'.format(ix)][sel, :].flatten()
                y_rot += m.trace['mu_rotcos{}'.format(ix)][sel, :].flatten()

        if 'orb' in modelcomponent:
            N_harmonics = int(modelcomponent[0])
            for ix in range(N_harmonics):
                y_orb += m.trace['mu_orbsin{}'.format(ix)][sel, :].flatten()
                y_orb += m.trace['mu_orbcos{}'.format(ix)][sel, :].flatten()

    # make the plot!
    plt.close('all')
    fig, axs = plt.subplots(nrows=4, figsize=(14, 12), sharex=True)

    axs[0].set_ylabel('flux')
    axs[0].plot(m.x_obs, m.y_obs, ".k", ms=4, label="data")
    axs[0].plot(m.x_obs, y_mod, lw=0.5, label='model',
                color='C0', alpha=1, zorder=5)

    for ix, f in enumerate(['rot', 'orb']):
        if f == 'rot':
            axs[0].plot(m.x_obs, y_rot, lw=0.5, label='model '+f,
                        color='C{}'.format(ix+1), alpha=1, zorder=ix+3)
        if f == 'orb':
            axs[0].plot(m.x_obs, y_orb+y_tra, lw=0.5, label='model '+f,
                        color='C{}'.format(ix+1), alpha=1, zorder=ix+3)

    axs[1].set_ylabel('flux-orb (rot)')
    axs[1].plot(m.x_obs, m.y_obs-y_orb-y_tra, ".k", ms=4, label="data-orb")
    axs[1].plot(m.x_obs, y_mod-y_orb-y_tra, lw=0.5,
                label='model-orb', color='C0', alpha=1, zorder=5)

    axs[2].set_ylabel('flux-rot (orb)')
    axs[2].plot(m.x_obs, m.y_obs-y_rot, ".k", ms=4, label="data-rot")
    axs[2].plot(m.x_obs, y_mod-y_rot, lw=0.5,
                label='model-rot', color='C0', alpha=1, zorder=5)

    axs[3].set_ylabel('flux-model')
    axs[3].plot(m.x_obs, m.y_obs-y_mod, ".k", ms=4, label="data")
    axs[3].plot(m.x_obs, y_mod-y_mod, lw=0.5, label='model',
                color='C0', alpha=1, zorder=5)

    axs[-1].set_xlabel("time [days]")
    for a in axs:
        a.legend()
        format_ax(a)
    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)

    ydict = {
        'x_obs': m.x_obs,
        'y_obs': m.y_obs,
        'y_orb': m.y_obs-y_rot,
        'y_rot': m.y_obs-y_orb,
        'y_mod_tra': y_tra,
        'y_mod_rot': y_orb,
        'y_mod_orb': y_rot
    }
    return ydict


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



def plot_traceplot(m, outpath):
    # trace plot from PyMC3
    if not os.path.exists(outpath):
        plt.figure(figsize=(7, 7))
        pm.traceplot(m.trace[100:])
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close('all')


def plot_cornerplot(true_d, m, outpath):

    if os.path.exists(outpath) and not m.OVERWRITE:
        return

    # corner plot of posterior samples
    plt.close('all')
    trace_df = trace_to_dataframe(m.trace, varnames=list(true_d.keys()))
    truths = [true_d[k] for k in true_d.keys()]
    truths = list(bflatten(truths))
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        truths=truths, title_fmt='.2g')
    savefig(fig, outpath, writepdf=0, dpi=100)


def plot_scene(c_obj, img_wcs, img, outpath, Tmag_cutoff=17, showcolorbar=0,
               ap_mask=0, bkgd_mask=0):
    #FIXME: will have to update for 837
    #FIXME: will have to update for 837
    #FIXME: will have to update for 837

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
    radius = 5.0*u.arcminute

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
    interval = vis.AsymmetricPercentileInterval(20,99)
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

    ax0.scatter(px, py, marker='x', c='C1', s=20, rasterized=True, zorder=3,
                linewidths=0.8)
    ax0.plot(target_x, target_y, mew=0.5, zorder=5, markerfacecolor='yellow',
             markersize=15, marker='*', color='k', lw=0)

    ax0.text(3.2, 5, 'A', fontsize=16, color='C1', zorder=6, style='italic')

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
