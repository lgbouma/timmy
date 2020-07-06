"""
Plots:

    plot_MAP_rv
    plot_quicklooklc

    plot_fitted_zoom
    plot_raw_zoom
    plot_fitindiv

    plot_phasefold
    plot_scene
    plot_hr
    plot_lithium
    plot_rotation
    plot_fpscenarios
    plot_grounddepth

    plot_full_kinematics
        (plot_positions)
        (plot_velocities)

    groundphot:
        plot_groundscene
        shift_img_plot
        plot_pixel_lc
        vis_photutils_lcs
        stackviz_blend_check

    convenience:
        hist2d
        plot_contour_2d_samples
"""
import os, corner, pickle
from datetime import datetime
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
from numpy import array as nparr
from scipy.interpolate import interp1d
from itertools import product

from billy.plotting import savefig, format_ax
import billy.plotting as bp

from timmy.paths import DATADIR, RESULTSDIR
from timmy.convenience import (
    get_tessphot, get_clean_tessphot, detrend_tessphot, get_model_transit,
    get_model_transit_quad, _get_fitted_data_dict,
    _get_fitted_data_dict_alltransit, _get_fitted_data_dict_allindivtransit
)


from astrobase.lcmath import (
    phase_magseries, phase_bin_magseries, sigclip_magseries,
    find_lc_timegroups, phase_magseries_with_errs
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

from scipy.ndimage import gaussian_filter
import logging
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects

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

def plot_MAP_rv(x_obs, y_obs, y_MAP, y_err, telcolors, x_pred, y_pred,
                map_estimate, outpath):

    #
    # rv vs time
    #
    plt.close('all')
    plt.figure(figsize=(14, 4))

    plt.plot(x_pred, y_pred, "k", lw=0.5)

    plt.errorbar(x_obs, y_MAP, yerr=y_err, fmt=",k")
    plt.scatter(x_obs, y_MAP, c=telcolors, s=8, zorder=100)

    plt.xlim(x_pred.min(), x_pred.max())
    plt.xlabel("BJD")
    plt.ylabel("radial velocity [m/s]")
    _ = plt.title("MAP model")

    fig = plt.gcf()
    savefig(fig, outpath, writepdf=0, dpi=300)

    outpath = outpath.replace('.png', '_phasefold.png')

    #
    # rv vs phase
    #
    plt.close('all')

    obs_d = phase_magseries_with_errs(
        x_obs, y_MAP, y_err, map_estimate['period'], map_estimate['t0'],
        wrap=False, sort=False
    )
    pred_d = phase_magseries(
        x_pred, y_pred, map_estimate['period'], map_estimate['t0'],
        wrap=False, sort=True
    )

    plt.plot(
        pred_d['phase'], pred_d['mags'], "k", lw=0.5
    )
    plt.errorbar(
        obs_d['phase'], obs_d['mags'], yerr=obs_d['errs'], fmt=",k"
    )
    plt.scatter(
        obs_d['phase'], obs_d['mags'], c=telcolors, s=8, zorder=100
    )

    plt.xlabel("phase")
    plt.ylabel("radial velocity [m/s]")
    _ = plt.title("MAP model. P={:.5f}, t0={:.5f}".
                  format(map_estimate['period'], map_estimate['t0']),
                  fontsize='small')

    fig = plt.gcf()
    savefig(fig, outpath, writepdf=0, dpi=300)


###############
# convenience #
###############
def hist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, quiet=False,
           plot_datapoints=False, plot_density=False,
           plot_contours=True, no_fill_contours=False, fill_contours=True,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
           pcolor_kwargs=None, **kwargs):
    """
    Plot a 2-D histogram of samples.
    (Function stolen from corner.py --  Foreman-Mackey, (2016), corner.py:
    Scatterplot matrices in Python, Journal of Open Source Software, 1(2), 24,
    doi:10.21105/joss.00024.)

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    quiet : bool
        If true, suppress warnings for small datasets.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    pcolor_kwargs : dict
        Any additional keyword arguments to pass to the `pcolor` method when
        adding the density colormap.

    """
    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range' instead.")
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2)
        # levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    # rgba_color = colorConverter.to_rgba(color)
    rgba_color = [0.0, 0.0, 0.0, 0.7]
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, range)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    if plot_contours or plot_density:
        # Compute the density levels.
        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        if np.any(m) and not quiet:
            logging.warning("Too few points to create valid contours")
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()

        # Compute the bin centers.
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

        # Extend the array for the sake of the contours at the plot edges.
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate([
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
        Y2 = np.concatenate([
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        if pcolor_kwargs is None:
            pcolor_kwargs = dict()
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap, **pcolor_kwargs)

    # Plot the contour edge colors.
    if plot_contours:
        cs = ax.contour(X2, Y2, H2.T, V, colors='k', linewidths=1, zorder=3)
        if kwargs['return_3sigma']:
            # index of 3 sigma line is, in this case, 0! and it's the vertices
            # along that contour that we often care about
            p = cs.collections[0].get_paths()[0]
            v = p.vertices
            x_3sig = v[:,0]
            y_3sig = v[:,1]

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])

    if not kwargs['return_3sigma']:
        return ax
    else:
        return ax, x_3sig, y_3sig


def plot_contour_2d_samples(xsample, ysample, xgrid, ygrid, outpath,
                            xlabel='logper', ylabel='logk',
                            return_3sigma=True, smooth=None):

    fig, ax = plt.subplots(figsize=(4,3))

    # smooth of 1.0 was ok
    bins = (xgrid, ygrid)

    ax, x_3sig, y_3sig = hist2d(
        xsample, ysample, bins=bins, range=None,
        weights=None, levels=None, smooth=smooth, ax=ax, color=None, quiet=False,
        plot_datapoints=False, plot_density=False, plot_contours=True,
        no_fill_contours=False, fill_contours=True, contour_kwargs=None,
        contourf_kwargs=None, data_kwargs=None, pcolor_kwargs=None,
        return_3sigma=return_3sigma
    )

    ax.plot(x_3sig, y_3sig, color='C0', zorder=5, lw=1)

    ax.set_xscale('linear')
    ax.set_yscale('linear')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout(h_pad=0, w_pad=0)

    format_ax(ax)

    savefig(fig, outpath)

    if return_3sigma:
        return x_3sig, y_3sig



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

    time, flux, flux_err = get_clean_tessphot(provenance, yval, binsize=None,
                                              maskflares=0)

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
                      titlestr=None, ylim=None):

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

        if 'Evans' in provenance:
            period = 8.3248972
            t0 = 2457000 + 1574.2738304
            tdur = 1.91/24

        if showvlines and provenance == 'Evans_2020-04-01':
            tra_ix = 44
        if showvlines and provenance == 'Evans_2020-04-26':
            tra_ix = 47
        if showvlines and provenance == 'Evans_2020-05-21':
            tra_ix = 50

        if 'Evans' in provenance:
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                [t0 + period*tra_ix - tdur/2, t0 + period*tra_ix + tdur/2],
                ymin, ymax, colors='C1', alpha=0.5, linestyles='--', zorder=-2,
                linewidths=0.5
            )
            ax.set_ylim((ymin, ymax))

        ax.set_xlim((xmin, xmax))

        format_ax(ax)

    if isinstance(ylim, tuple):
        axs[1].set_ylim(ylim)

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

    time, flux, flux_err = get_clean_tessphot(provenance, yval, binsize=None,
                                              maskflares=0)
    flat_flux, trend_flux = detrend_tessphot(time, flux, flux_err)
    if detrend:
        flux = flat_flux

    t_offset = np.nanmin(time)
    time -= t_offset

    FLARETIMES = [
        (4.60, 4.63),
        (37.533, 37.62)
    ]
    flaresel = np.zeros_like(time).astype(bool)
    for ft in FLARETIMES:
        flaresel |= ( (time > min(ft)) & (time < max(ft)) )

    t0 = 1574.2738 - t_offset
    per = 8.32467
    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    plt.close('all')

    ##########################################

    # figsize=(8.5, 10) full page... 10 leaves space.
    fig = plt.figure(figsize=(8.5, 5))

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
    yval = (flux - np.nanmedian(flux))*1e3
    ax0.scatter(time, yval, c='k', zorder=3, s=0.75, rasterized=True,
                linewidths=0)
    ax0.scatter(time[flaresel], yval[flaresel], c='r', zorder=3, s=1,
                rasterized=True, linewidths=0)
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
        ax.scatter(time[s], (flux[s] - np.nanmedian(flux[s]))*1e3, c='k',
                   zorder=3, s=7, rasterized=False, linewidths=0)

        _flaresel = np.zeros_like(time[s]).astype(bool)
        for ft in FLARETIMES:
            _flaresel |= ( (time[s] > min(ft)) & (time[s] < max(ft)) )

        if np.any(_flaresel):
            ax.scatter(time[s][_flaresel],
                       (flux[s] - np.nanmedian(flux[s]))[_flaresel]*1e3,
                       c='r', zorder=3, s=8, rasterized=False, linewidths=0)

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
    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center',
             rotation=90, fontsize='x-large')

    fig.tight_layout(h_pad=0.2, w_pad=-0.5)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_phasefold(m, summdf, outpath, overwrite=0, show_samples=0,
                   modelid=None, inppt=0, showerror=1):

    if modelid is None:
        d, params, paramd = _get_fitted_data_dict(m, summdf)
        _d = d

    elif 'alltransit' in modelid:
        d = _get_fitted_data_dict_alltransit(m, summdf)
        _d = d['tess']

    elif 'allindivtransit' in modelid:
        d = _get_fitted_data_dict_allindivtransit(m, summdf)
        _d = d['tess']

    P_orb = summdf.loc['period', 'median']
    t0_orb = summdf.loc['t0', 'median']

    # phase and bin them.
    binsize = 5e-4
    orb_d = phase_magseries(
        _d['x_obs'], _d['y_obs'], P_orb, t0_orb, wrap=True, sort=True
    )
    orb_bd = phase_bin_magseries(
        orb_d['phase'], orb_d['mags'], binsize=binsize, minbinelems=3
    )
    mod_d = phase_magseries(
        _d['x_obs'], _d['y_mod'], P_orb, t0_orb, wrap=True, sort=True
    )
    resid_bd = phase_bin_magseries(
        mod_d['phase'], orb_d['mags'] - mod_d['mags'], binsize=binsize,
        minbinelems=3
    )

    # get the samples. shape: N_samples x N_time
    if show_samples:
        np.random.seed(42)
        N_samples = 20

        sample_df = pm.trace_to_dataframe(m.trace, var_names=params)
        sample_params = sample_df.sample(n=N_samples, replace=False)

        y_mod_samples = []
        for ix, p in sample_params.iterrows():
            print(ix)
            paramd = dict(p)
            y_mod_samples.append(get_model_transit(paramd, d['x_obs']))

        y_mod_samples = np.vstack(y_mod_samples)

        mod_ds = {}
        for i in range(N_samples):
            mod_ds[i] = phase_magseries(
                d['x_obs'], y_mod_samples[i, :], P_orb, t0_orb, wrap=True,
                sort=True
            )

    # make tha plot
    plt.close('all')

    fig, (a0, a1) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                 figsize=(4, 3), gridspec_kw=
                                 {'height_ratios':[3, 2]})

    if not inppt:

        a0.scatter(orb_d['phase']*P_orb*24, orb_d['mags'], color='gray', s=2,
                   alpha=0.8, zorder=4, linewidths=0, rasterized=True)
        a0.scatter(orb_bd['binnedphases']*P_orb*24, orb_bd['binnedmags'],
                   color='black', s=8, alpha=1, zorder=5, linewidths=0)
        a0.plot(mod_d['phase']*P_orb*24, mod_d['mags'], color='darkgray',
                alpha=0.8, rasterized=False, lw=1, zorder=1)

        a1.scatter(orb_d['phase']*P_orb*24, orb_d['mags']-mod_d['mags'],
                   color='gray', s=2, alpha=0.8, zorder=4, linewidths=0,
                   rasterized=True)
        a1.scatter(resid_bd['binnedphases']*P_orb*24, resid_bd['binnedmags'],
                   color='black', s=8, alpha=1, zorder=5, linewidths=0)
        a1.plot(mod_d['phase']*P_orb*24, mod_d['mags']-mod_d['mags'],
                color='darkgray', alpha=0.8, rasterized=False, lw=1, zorder=1)

    else:

        ydiff = 0 if modelid == 'allindivtransit' else 1

        a0.scatter(orb_d['phase']*P_orb*24, 1e3*(orb_d['mags']-ydiff),
                   color='darkgray', s=7, alpha=0.5, zorder=4, linewidths=0,
                   rasterized=True)
        a0.scatter(orb_bd['binnedphases']*P_orb*24,
                   1e3*(orb_bd['binnedmags']-ydiff), color='black', s=18, alpha=1,
                   zorder=5, linewidths=0)
        a0.plot(mod_d['phase']*P_orb*24, 1e3*(mod_d['mags']-ydiff),
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=1)

        a1.scatter(orb_d['phase']*P_orb*24, 1e3*(orb_d['mags']-mod_d['mags']),
                   color='darkgray', s=7, alpha=0.5, zorder=4, linewidths=0,
                   rasterized=True)
        a1.scatter(resid_bd['binnedphases']*P_orb*24,
                   1e3*resid_bd['binnedmags'], color='black', s=18, alpha=1,
                   zorder=5, linewidths=0)
        a1.plot(mod_d['phase']*P_orb*24, 1e3*(mod_d['mags']-mod_d['mags']),
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=1)


    if show_samples:
        # NOTE: this comes out looking "bad" because if you phase up a model
        # with a different period to the data, it will produce odd
        # aliases/spikes.

        xvals, yvals = [], []
        for i in range(N_samples):
            xvals.append(mod_ds[i]['phase']*P_orb*24)
            yvals.append(mod_ds[i]['mags'])
            a0.plot(mod_ds[i]['phase']*P_orb*24, mod_ds[i]['mags'], color='C1',
                    alpha=0.2, rasterized=True, lw=0.2, zorder=-2)
            a1.plot(mod_ds[i]['phase']*P_orb*24,
                    mod_ds[i]['mags']-mod_d['mags'], color='C1', alpha=0.2,
                    rasterized=True, lw=0.2, zorder=-2)

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

    if not inppt:
        a0.set_ylabel('Relative flux', fontsize='small')
    else:
        a0.set_ylabel('Relative flux [ppt]', fontsize='small')
    a1.set_ylabel('Residual [ppt]', fontsize='small')
    a1.set_xlabel('Hours from mid-transit', fontsize='small')

    if not inppt:
        a0.set_ylim((0.9925, 1.005))

    yv = orb_d['mags']-mod_d['mags']
    if inppt:
        yv = 1e3*(orb_d['mags']-mod_d['mags'])
    a1.set_ylim((np.nanmedian(yv)-3.2*np.nanstd(yv),
                 np.nanmedian(yv)+3.2*np.nanstd(yv) ))

    for a in (a0, a1):
        a.set_xlim((-0.011*P_orb*24, 0.011*P_orb*24))
        # a.set_xlim((-0.02*P_orb*24, 0.02*P_orb*24))
        format_ax(a)

    if inppt:
        a0.set_ylim((-6.9, 4.1))
        for a in (a0, a1):
            xval = np.arange(-2,3,1)
            a.set_xticks(xval)
        yval = np.arange(-5,5,2.5)
        a0.set_yticks(yval)

    if showerror:
        trans = transforms.blended_transform_factory(
                a0.transAxes, a0.transData)
        if inppt:
            _e = 1e3*np.median(_d['y_err'])
            # a0.errorbar(
            #     0.9, -5, yerr=_e, fmt='none',
            #     ecolor='darkgray', alpha=0.8, elinewidth=1, capsize=2,
            #     transform=trans
            # )

            # bin to roughly 5e-4 * 8.3 * 24 * 60 ~= 6 minute intervals
            bintime = binsize*P_orb*24*60
            sampletime = 2 # minutes
            errorfactor = (sampletime/bintime)**(1/2)

            a0.errorbar(
                0.85, -5, yerr=errorfactor*_e,
                fmt='none', ecolor='black', alpha=1, elinewidth=1, capsize=2,
                transform=trans
            )

            print(f'{_e:.2f}, {errorfactor*_e:.2f}')

        else:
            raise NotImplementedError

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
                        linewidth=0., zorder=2, alpha=1, rasterized=True,
                        color='white'
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

    ax0.scatter(px, py, marker='o', c='white', s=1.5*5e4/(tmags**3),
                rasterized=False, zorder=6, linewidths=0.5, edgecolors='k')
    # ax0.scatter(px, py, marker='x', c='C1', s=20, rasterized=True,
    #             zorder=6, linewidths=0.8)
    ax0.plot(target_x, target_y, mew=0.5, zorder=5,
             markerfacecolor='yellow', markersize=18, marker='*',
             color='k', lw=0)

    t = ax0.text(4.0, 5.2, 'A', fontsize=20, color='k', zorder=6)#, style='italic')
    t.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground='white'),
                        path_effects.Normal()])
    t = ax0.text(4.6, 3.8, 'B', fontsize=20, color='k', zorder=6)#, style='italic')
    t.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground='white'),
                        path_effects.Normal()])

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

    # # DSS is ~1 arcsecond per pixel. overplot apertures on axes 6,7
    # for ix, radius_px in enumerate([21,21*1.5,21*2.25]):
    #     circle = plt.Circle((sizepix/2, sizepix/2), radius_px,
    #                         color='C{}'.format(ix), fill=False, zorder=5+ix)
    #     ax1.add_artist(circle)

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
        markersize=14, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    # ax.set_ylabel('G + $5\log_{10}(\omega_{\mathrm{as}}) + 5$', fontsize='large')
    # ax.set_xlabel('Bp - Rp', fontsize='large')
    ax.set_ylabel('Absolute G [mag]', fontsize='large')
    ax.set_xlabel('Bp - Rp [mag]', fontsize='large')

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

    f, axs = plt.subplots(figsize=(6,6), nrows=nparams-1, ncols=nparams-1)

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
                target_df[xv], target_df[yv], alpha=1, mew=0.5,
                zorder=8, label='TOI 837', markerfacecolor='yellow',
                markersize=14, marker='*', color='black', lw=0
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

    N_trim = 47 # per Phil Evan's reduction notes
    if '2020-04-01' not in outpath:
        raise NotImplementedError(
            'pixel LCs are deprecated. only 2020-04-01 was implemented'
        )

    for ax_i, data_i in enumerate(range(xmin, xmax)):
        for ax_j, data_j in enumerate(list(range(ymin, ymax))[::-1]):
           # note: y reverse is the same as "origin = lower"

            print(ax_i, ax_j, data_i, data_j)

            axs[ax_j,ax_i].scatter(
                times[N_trim:], img_cube[N_trim:, data_j, data_i], c='k', zorder=3, s=2,
                rasterized=True, linewidths=0
            )

            tstr = (
                '{:.1f}\n{} {}'.format(
                    np.nanpercentile( img_cube[N_trim:, data_j, data_i], 99),
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

    if datestr == '2020-04-01':
        N_trim = 47 # drop the first 47 points due to clouds
    elif datestr == '2020-04-26':
        N_trim = 0
    elif datestr == '2020-05-21':
        N_trim = 0
    else:
        raise NotImplementedError('pls manually set N_trim')

    time = target_lc['BJD_TDB'][N_trim:]
    flux = target_lc[ap][N_trim:]
    mean_flux = np.nanmean(flux)
    flux /= mean_flux

    print(42*'-')
    print(target_ticid, target_lc.id.iloc[0], mean_flux)

    comp_mean_fluxs = nparr([np.nanmean(lc[ap][N_trim:]) for lc in lcs])
    comp_inds = np.argsort(np.abs(mean_flux - comp_mean_fluxs))

    # the maximum number of comparison stars is say, 10.
    N_comp_max = 10

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
    for ix, comp_ind in enumerate(comp_inds[1:N_comp_max+1]):

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
        outdf['absflux_{}'.format(t)] = flux*mean_flux

    outcsvpath = os.path.join(
        RESULTSDIR,  'groundphot', datestr, 'vis_photutils_lcs',
        'vis_photutils_lcs_compstars_{}.csv'.format(ap)
    )

    outdf.to_csv(outcsvpath, index=False)
    print('made {}'.format(outcsvpath))

    format_ax(ax)

    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)


def stackviz_blend_check(datestr, apn, soln=0, overwrite=1, adaptiveoffset=1,
                         N_comp=5):

    if soln == 1:
        raise NotImplementedError('gotta implement image+aperture inset axes')

    if adaptiveoffset:
        outdir = os.path.join(
            RESULTSDIR,  'groundphot', datestr,
            f'stackviz_blend_check_adaptiveoffset_Ncomp{N_comp}'
        )
    else:
        outdir = os.path.join(
            RESULTSDIR,  'groundphot', datestr,
            f'stackviz_blend_check_noadaptiveoffset_Ncomp{N_comp}'
        )
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'stackviz_blend_check_{}.png'.format(apn))

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    lcdir = f'/Users/luke/Dropbox/proj/timmy/results/groundphot/{datestr}/compstar_detrend_Ncomp{N_comp}'
    lcpaths = np.sort(glob(os.path.join(
        lcdir, 'toi837_detrended*_sum_{}_*.csv'.format(apn))))
    assert len(lcpaths) == 13

    origlcdir = f'/Users/luke/Dropbox/proj/timmy/results/groundphot/{datestr}/vis_photutils_lcs'

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
        txt_x, txt_y = (
            np.nanpercentile(time, 97), np.nanpercentile(flux+offset, 1)
        )
        if adaptiveoffset:
            ax.text(txt_x, txt_y,
                    tstr, ha='right', va='bottom', bbox=props, zorder=-1,
                    fontsize='small', color=color)
        else:
            ax.text(txt_x, max(txt_y, 0.9),
                    tstr, ha='right', va='bottom', bbox=props, zorder=-1,
                    fontsize='small', color=color)

        if adaptiveoffset:
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
        else:
            offset += 0.10

    if not adaptiveoffset:
        ax.set_ylim((0.9, 2.3))


    format_ax(ax)

    fig.tight_layout()
    savefig(fig, outpath, writepdf=0, dpi=300)


def plot_fitted_zoom(m, summdf, outpath, overwrite=1, modelid=None):

    yval = "PDCSAP_FLUX"
    provenance = 'spoc'
    detrend = 1

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    if modelid is None:
        d, params, _ = _get_fitted_data_dict(m, summdf)
        _d = d
    elif 'alltransit' in modelid:
        d = _get_fitted_data_dict_alltransit(m, summdf)
        _d = d['tess']

    time, flux, flux_err = _d['x_obs'], _d['y_obs'], _d['y_err']

    t_offset = np.nanmin(time)
    time -= t_offset

    t0 = summdf.loc['t0', 'median'] - t_offset
    per = summdf.loc['period', 'median']

    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    plt.close('all')

    ##########################################

    # figsize=(8.5, 10) full page... 10 leaves space.
    fig = plt.figure(figsize=(8.5*1.5, 8))

    ax0 = plt.subplot2grid(shape=(3,5), loc=(0,0), colspan=5)

    ax1 = plt.subplot2grid((3,5), (1,0), colspan=1)
    ax2 = plt.subplot2grid((3,5), (1,1), colspan=1)
    ax3 = plt.subplot2grid((3,5), (1,2), colspan=1)
    ax4 = plt.subplot2grid((3,5), (1,3), colspan=1)
    ax5 = plt.subplot2grid((3,5), (1,4), colspan=1)

    ax6 = plt.subplot2grid((3,5), (2,0), colspan=1)
    ax7 = plt.subplot2grid((3,5), (2,1), colspan=1)
    ax8 = plt.subplot2grid((3,5), (2,2), colspan=1)
    ax9 = plt.subplot2grid((3,5), (2,3), colspan=1)
    ax10 = plt.subplot2grid((3,5), (2,4), colspan=1)

    all_axs = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]
    tra_axs = [ax1,ax2,ax3,ax4,ax5]
    res_axs = [ax6,ax7,ax8,ax9,ax10]
    tra_ixs = [0,2,3,4,5]

    # main lightcurve
    yval = (flux - np.nanmean(flux))*1e3
    ax0.scatter(time, yval, c='k', zorder=3, s=0.75, rasterized=True,
                linewidths=0)
    ax0.plot(time, (_d['y_mod'] - np.nanmean(flux))*1e3, color='C0', alpha=0.8,
             rasterized=False, lw=1, zorder=4)

    ax0.set_ylim((-20, 20)) # omitting like 1 upper point from the big flare at time 38
    ymin, ymax = ax0.get_ylim()
    ax0.vlines(
        tra_times, ymin, ymax, colors='C1', alpha=0.5,
        linestyles='--', zorder=-2, linewidths=0.5
    )
    ax0.set_ylim((ymin, ymax))
    ax0.set_xlim((np.nanmin(time)-1, np.nanmax(time)+1))

    # zoom-in of raw transits
    for ax, tra_ix, rax in zip(tra_axs, tra_ixs, res_axs):

        mid_time = t0 + per*tra_ix
        tdur = 2/24. # roughly, in units of days
        # n = 2.5 # good
        n = 2.0 # good
        start_time = mid_time - n*tdur
        end_time = mid_time + n*tdur

        s = (time > start_time) & (time < end_time)
        ax.scatter(time[s], (flux[s] - np.nanmean(flux[s]))*1e3, c='k',
                   zorder=3, s=7, rasterized=False, linewidths=0)
        ax.plot(time[s], (_d['y_mod'][s] - np.nanmean(flux[s]))*1e3 ,
                color='C0', alpha=0.8, rasterized=False, lw=1, zorder=1)

        rax.scatter(time[s], (flux[s] - _d['y_mod'][s])*1e3, c='k',
                    zorder=3, s=7, rasterized=False, linewidths=0)
        rax.plot(time[s], (_d['y_mod'][s] - _d['y_mod'][s])*1e3, color='C0',
                 alpha=0.8, rasterized=False, lw=1, zorder=1)

        ax.set_ylim((-8, 8))
        rax.set_ylim((-8, 8))

        for a in [ax,rax]:
            a.set_xlim((start_time, end_time))

            ymin, ymax = a.get_ylim()
            a.vlines(
                mid_time, ymin, ymax, colors='C1', alpha=0.5,
                linestyles='--', zorder=-2, linewidths=0.5
            )
            a.set_ylim((ymin, ymax))


            if tra_ix > 0:
                # hide the ytick labels
                labels = [item.get_text() for item in
                          a.get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                a.set_yticklabels(empty_string_labels)

        labels = [item.get_text() for item in
                  ax.get_yticklabels()]
        empty_string_labels = ['']*len(labels)
        ax.set_xticklabels(empty_string_labels)

    for ax in all_axs:
        format_ax(ax)

    fig.text(0.5,-0.01, 'Time [days]', ha='center', fontsize='x-large')
    fig.text(-0.01,0.5, 'Relative flux [part per thousand]', va='center',
             rotation=90, fontsize='x-large')

    fig.tight_layout(h_pad=0.2, w_pad=-1.0)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_lithium(outdir):

    from timmy.lithium import get_Randich18_lithium, get_Berger18_lithium

    rdf = get_Randich18_lithium()
    bdf = get_Berger18_lithium()

    selclusters = [
        # 'IC4665', # LDB 23.2 Myr
        'NGC2547', # LDB 37.7 Myr
        'IC2602', # LDB 43.7 Myr
        # 'IC2391', # LDB 51.3 Myr
    ]
    selrdf = np.zeros(len(rdf)).astype(bool)
    for c in selclusters:
        selrdf |= rdf.Cluster.str.contains(c)

    srdf = rdf[selrdf]
    srdf_lim = srdf[srdf.f_EWLi==3]
    srdf_val = srdf[srdf.f_EWLi==0]

    # young dictionary
    yd = {
        'val_teff_young': nparr(srdf_val.Teff),
        'val_teff_err_young': nparr(srdf_val.e_Teff),
        'val_li_ew_young': nparr(srdf_val.EWLi),
        'val_li_ew_err_young': nparr(srdf_val.e_EWLi),
        'lim_teff_young': nparr(srdf_lim.Teff),
        'lim_teff_err_young': nparr(srdf_lim.e_Teff),
        'lim_li_ew_young': nparr(srdf_lim.EWLi),
        'lim_li_ew_err_young': nparr(srdf_lim.e_EWLi),
    }

    # field dictionary
    # SNR > 3
    field_det = ( (bdf.EW_Li_ / bdf.e_EW_Li_) > 3 )
    bdf_val = bdf[field_det]
    bdf_lim = bdf[~field_det]

    fd = {
        'val_teff_field': nparr(bdf_val.Teff),
        'val_li_ew_field': nparr(bdf_val.EW_Li_),
        'val_li_ew_err_field': nparr(bdf_val.e_EW_Li_),
        'lim_teff_field': nparr(bdf_lim.Teff),
        'lim_li_ew_field': nparr(bdf_lim.EW_Li_),
        'lim_li_ew_err_field': nparr(bdf_lim.e_EW_Li_),
    }

    d = {**yd, **fd}

    ##########
    # make tha plot 
    ##########

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    classes = ['young', 'field']
    colors = ['k', 'gray']
    zorders = [2, 1]
    markers = ['o', '.']
    labels = ['NGC$\,$2547 & IC$\,$2602', 'Kepler Field']

    # plot vals
    for _cls, _col, z, m, l in zip(classes, colors, zorders, markers, labels):
        ax.scatter(
            d[f'val_teff_{_cls}'], d[f'val_li_ew_{_cls}'], c=_col, alpha=1,
            zorder=z, s=5, rasterized=False, linewidths=0, label=l, marker=m
        )


    from timmy.priors import TEFF, LI_EW
    ax.plot(
        TEFF,
        LI_EW,
        alpha=1, mew=0.5, zorder=8, label='TOI 837', markerfacecolor='yellow',
        markersize=14, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    ax.set_ylabel('Li$_{6708}$ EW [m$\mathrm{\AA}$]', fontsize='large')
    ax.set_xlabel('Effective Temperature [K]', fontsize='large')

    ax.set_xlim((4900, 6600))

    format_ax(ax)
    outpath = os.path.join(outdir, 'lithium.png')
    savefig(f, outpath)


def plot_rotation(outdir):

    from timmy.paths import DATADIR
    rotdir = os.path.join(DATADIR, 'rotation')

    # make plot
    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    classes = ['pleiades', 'praesepe', 'ngc6811']
    colors = ['k', 'gray', 'darkgray']
    zorders = [3, 2, 1]
    markers = ['o', 'X', 's']
    labels = ['Pleaides', 'Praesepe', 'NGC$\,$6811']

    # plot vals
    for _cls, _col, z, m, l in zip(classes, colors, zorders, markers, labels):
        df = pd.read_csv(os.path.join(rotdir, f'curtis19_{_cls}.csv'))
        ax.scatter(
            df['teff'], df['prot'], c=_col, alpha=1, zorder=z, s=5,
            rasterized=False, linewidths=0, label=l, marker=m
        )


    from timmy.priors import TEFF, P_ROT
    ax.plot(
        TEFF,
        P_ROT,
        alpha=1, mew=0.5, zorder=8, label='TOI 837', markerfacecolor='yellow',
        markersize=14, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    ax.set_ylabel('Rotation Period [days]', fontsize='large')
    ax.set_xlabel('Effective Temperature [K]', fontsize='large')

    ax.set_xlim((4900, 6600))
    ax.set_ylim((0,14))

    format_ax(ax)
    outpath = os.path.join(outdir, 'rotation.png')
    savefig(f, outpath)


def _get_color_df(tdepth_ap, sep_arcsec, fn_mass_to_dmag, band='Rc'):

    color_path = os.path.join(RESULTSDIR, 'fpscenarios', f'multicolor_{band}.csv')
    color_data_df = pd.read_csv(color_path)

    m2_color = max(color_data_df[color_data_df['frac_viable'] == 0].m2)

    color_ap = tdepth_ap-0.20    # arcsec, same as tdepth
    color_sep = sep_arcsec[sep_arcsec < color_ap]

    color_dmag = np.ones_like(color_sep)*fn_mass_to_dmag(m2_color)

    color_df = pd.DataFrame({'sep_arcsec': color_sep, 'dmag': color_dmag})
    _append_df = pd.DataFrame({
        'sep_arcsec':[2.00001],
        'dmag':[10]
    })
    color_df = color_df.append(_append_df)

    return color_df





def plot_fpscenarios(outdir):

    #
    # get data
    #

    #
    # speckle AO from SOAR HRcam
    #
    speckle_path = os.path.join(DATADIR, 'speckle', 'sep_vs_dmag.csv')
    speckle_df = pd.read_csv(speckle_path, names=['sep_arcsec','dmag'])

    dist_pc = 142.488 # pc, TIC8
    sep_arcsec = np.logspace(-2, 1.5, num=1000, endpoint=True)
    sep_au = sep_arcsec*dist_pc

    #
    # transit depth constraint: within 2 arcseconds from ground-based seeing
    # limited resolution.
    # within dmag~5.2 from transit depth (and assumption of a totally eclipsing
    # M+M dwarf type scenario. Totally eclipsing dark companion+Mdwarf is a bit
    # ridiculous).
    #
    N = 0.5 # N=1 for the dark companion scenario.
    Tmag = 9.9322
    depth_obs = (4374e-6) # QLP depth
    tdepth_ap = 2    # arcsec
    tdepth_sep = sep_arcsec[sep_arcsec < tdepth_ap]
    tdepth_dmag = 5/2*np.log10(N/depth_obs)*np.ones_like(tdepth_sep)
    tdepth_sep = np.append(tdepth_sep, tdepth_ap)
    tdepth_dmag = np.append(tdepth_dmag, 0)
    tdepth_df = pd.DataFrame({'sep_arcsec': tdepth_sep, 'dmag': tdepth_dmag})
    _append_df = pd.DataFrame({
        'sep_arcsec':[2.00001, 10],
        'dmag':[-1, -1]
    })
    tdepth_df = tdepth_df.append(_append_df)

    #
    # no double lined SB2 constraint. Zhou quoted F2/F1 ~= 2% as
    # limit. use 5%.
    #
    outer_lim = 1.0  # arcsec
    sb2_sep = sep_arcsec[sep_arcsec < outer_lim]
    flux_frac = 5e-2
    dmag = -5/2 * np.log10(flux_frac)
    sb2_dmag = dmag * np.ones_like(sb2_sep)
    sb2_sep = np.append(sb2_sep, outer_lim)
    sb2_dmag = np.append(sb2_dmag, 0)
    sb2_df = pd.DataFrame({'sep_arcsec': sb2_sep, 'dmag': sb2_dmag})

    #
    # RV secondary radvel fitting, from drivers.calc_rvoutersensitivity
    #
    rv_path = os.path.join(RESULTSDIR, 'fpscenarios',
                           'rvoutersensitivity_3sigma.csv')
    rv_df = pd.read_csv(rv_path)
    rv_df['sma_au'] = 10**(rv_df.log10sma)
    rv_df['mp_msun'] = 10**(rv_df.log10mpsini)
    rv_df['sep_arcsec'] = rv_df['sma_au'] / dist_pc

    # conversion to contrast, from drivers.contrast_to_masslimit
    smooth_path = os.path.join(DATADIR, 'speckle', 'smooth_dmag_to_mass.csv')
    smooth_df = pd.read_csv(smooth_path)
    sel = ~pd.isnull(smooth_df['m_comp/m_sun'])
    smooth_df = smooth_df[sel]

    fn_mass_to_dmag = interp1d(
        nparr(smooth_df['m_comp/m_sun']), nparr(smooth_df['dmag_smooth']),
        kind='quadratic', bounds_error=False, fill_value=np.nan
    )

    rv_df['dmag'] = fn_mass_to_dmag(nparr(rv_df.mp_msun))
    sel = (
        (rv_df.sep_arcsec > 1e-3)
        &
        (rv_df.sep_arcsec < 1e1)
    )
    srv_df = rv_df[sel]

    # set the point one above the last finite dmag value to zero.
    srv_df.loc[np.nanargmin(nparr(srv_df.dmag))+1, 'dmag'] = 0

    #
    # color constraint
    #
    color_df_Rc = _get_color_df(
        tdepth_ap, sep_arcsec, fn_mass_to_dmag, band='Rc'
    )
    color_df_B = _get_color_df(
        tdepth_ap, sep_arcsec, fn_mass_to_dmag, band='B'
    )

    #
    # gaia constraint
    #
    gaia_path = os.path.join(DATADIR, 'gaia_contrast', 'gaia_contrast.csv')
    gaia_df = pd.read_csv(gaia_path)
    gaia_df['sep_arcsec'] = gaia_df['sep_milliarcsec']/1000
    _append_df = pd.DataFrame({
        'sep_arcsec':[max(gaia_df.sep_arcsec)],
        'dmag':[-1]
    })
    gaia_df = gaia_df.append(_append_df)

    ##########################################
    # make plot

    from timmy.multicolor import DELTA_LIM_RC, DELTA_LIM_B
    names = ['Transit depth', 'Speckle imaging', 'Not SB2', 'RVs',
             '$\delta_{R_C}>'+f'{1e3*DELTA_LIM_RC:.2f}'+'\,$ppt',
             '$\delta_{B_J}>'+f'{1e3*DELTA_LIM_B:.2f}'+'\,$ppt',
             'Gaia'
            ]
    sides = ['below', 'above', 'above', 'above', 'below', 'below', 'above']
    constraint_dfs = [tdepth_df, speckle_df, sb2_df, srv_df, color_df_Rc,
                      color_df_B, gaia_df]
    which = ['both', 'both', 'both', 'assoc', 'assoc', 'assoc', 'both']

    # colors = [f'C{ix}' for ix in range(len(constraint_dfs))]
    N_constraint = len(constraint_dfs)
    colors = plt.cm.YlGnBu(np.linspace(0.2,1,N_constraint))

    plt.close('all')

    f, axs = plt.subplots(nrows=2, ncols= 1, figsize=(4.5,6))

    for ax_ix, ax in enumerate(axs):
        for cdf, name, side, w, c in zip(
            constraint_dfs, names, sides, which, colors
        ):

            xval = cdf.sep_arcsec * dist_pc if ax_ix == 0 else cdf.sep_arcsec
            yval = cdf.dmag

            dofill = False
            if w == 'both':
                ax.plot(xval, yval, label=name, color=c, lw=3)
                dofill = True

            if w == 'assoc' and ax_ix == 0:
                ax.plot(xval, yval, label=name, color=c, lw=3)
                dofill = True
            elif w == 'assoc' and ax_ix == 1:
                pass

            if dofill:
                _c = 'gray'
                _a = 0.8
                if side == 'above':
                    ax.fill_between(
                        xval, yval, 0, color=_c, alpha=_a, lw=0
                    )
                elif side == 'below':
                    ax.fill_between(
                        xval, 7, yval, color=_c, alpha=_a, lw=0
                    )

    axs[0].set_title('Associated companions')
    axs[0].set_ylabel('Brightness contrast ($\Delta$mag)')
    axs[0].set_xlabel('Projected separation [AU]')

    axs[1].set_title('Chance alignments')
    axs[1].set_ylabel('Brightness contrast ($\Delta$mag)')
    axs[1].set_xlabel('Projected separation [arcsec]')

    for ax_ix, ax in enumerate(axs):
        ax.set_xscale('log')
        format_ax(ax)
        ax.set_ylim((7, 0))
        if ax_ix == 0:
            ax.set_xlim((1e-2*dist_pc, 1e1*dist_pc))
        else:
            ax.set_xlim((1e-2, 1e1))

    # Add the twin ax.  What you really want is something that is 1-1 between
    # the values on the y axis, (dmag 0-7), and the mass. This does that.
    tax = axs[0].twinx()
    tax.set_ylabel('Companion mass [M$_\odot$]')

    fn_dmag_to_mass = interp1d(
        nparr(smooth_df['dmag_smooth']), nparr(smooth_df['m_comp/m_sun']),
        kind='quadratic', bounds_error=False, fill_value=np.nan
    )

    mass_labels = fn_dmag_to_mass(np.arange(0,8,1))
    mass_labels = [f"{l:.2f}" for l in mass_labels][::-1]

    yval = np.arange(0,8,1)
    tax.plot(np.ones_like(yval), yval, c='k')

    tax.set_xscale('log')
    tax.set_yscale('linear')
    tax.set_xlim((1e-2*dist_pc, 1e1*dist_pc))
    tax.set_ylim([0, 7])

    tax.set_yticks(yval)
    tax.set_yticklabels(mass_labels)

    tax.yaxis.set_ticks_position('right')
    tax.get_yaxis().set_tick_params(which='both', direction='in')
    for tick in tax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')

    # Add the legend
    handles, labels = axs[0].get_legend_handles_labels()
    # lgd = axs[0].legend(handles, labels,
    #                 bbox_to_anchor=(2.0,-0.1))

    lgd = f.legend(
        handles=handles,
        labels=labels,
        loc='center left', borderaxespad=0.1,
        bbox_to_anchor=(0.92,0.46)
    )

    f.tight_layout()

    outpath = os.path.join(outdir, 'fpscenarios.png')

    # explicit savefig because legend is passed.
    fig=f
    figpath=outpath
    writepdf=True
    dpi=450

    fig.savefig(figpath, dpi=dpi, bbox_inches='tight', bbox_extra_artists=(lgd,))

    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))

    if writepdf:
        pdffigpath = figpath.replace('.png','.pdf')
        fig.savefig(pdffigpath, bbox_inches='tight', rasterized=True, dpi=dpi,
                    bbox_extra_artists=(lgd,))
        print('{}: made {}'.format(datetime.utcnow().isoformat(), pdffigpath))

    plt.close('all')


def plot_grounddepth(m, summdf, outpath, overwrite=1, modelid=None, showerror=1):

    from timmy.convenience import get_elsauce_phot, get_model_transit
    from copy import deepcopy
    from timmy.multicolor import DELTA_LIM_RC, DELTA_LIM_B
    from astrobase.lcmath import time_bin_magseries

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    if modelid is None:
        d, params, _ = _get_fitted_data_dict(m, summdf)
    elif 'alltransit' in modelid:
        d = _get_fitted_data_dict_alltransit(m, summdf)
    elif 'allindivtransit' in modelid:
        d = _get_fitted_data_dict_allindivtransit(m, summdf)

    ##########################################

    plt.close('all')

    fig, axs = plt.subplots(figsize=(4,3), ncols=2, nrows=2, sharex=True)

    tra_axs = axs.flatten()
    tra_ixs = [44, 47, 50, 53]

    titles = ['2020.04.01 R$_\mathrm{C}$',
              '2020.04.26 R$_\mathrm{C}$',
              '2020.05.21 I$_\mathrm{C}$',
              '2020.06.14 B$_\mathrm{J}$']

    datestrs = ['20200401', '20200426', '20200521', '20200614' ]

    inds = range(len(datestrs))

    t0 = summdf.loc['t0', 'median']
    per = summdf.loc['period', 'median']
    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    ##########################################
    for ind, ax, tra_ix, t, d in zip(inds, tra_axs, tra_ixs, titles, datestrs):

        ##########################################
        # get quantities to be plotted

        gtime, gflux, gflux_err = get_elsauce_phot(datestr=d)
        gtime = gtime - 2457000 # convert to BTJD

        gmodtime = np.linspace(np.nanmin(gtime)-1, np.nanmax(gtime)+1, int(1e4))
        if modelid is None:
            params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]', 'mean',
                      'r_star', 'logg_star']
        elif modelid == 'alltransit':
            params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]',
                      f'elsauce_{ind}_mean', 'r_star', 'logg_star']
        elif modelid in ['alltransit_quad', 'allindivtransit']:
            params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]',
                      f'elsauce_{ind}_mean', f'elsauce_{ind}_a1',
                      f'elsauce_{ind}_a2', 'r_star', 'logg_star']

        paramd = {k:summdf.loc[k, 'median'] for k in params}
        if modelid not in ['alltransit_quad', 'allindivtransit']:
            gmodflux = get_model_transit(paramd, gmodtime)
        else:
            _tmid = np.nanmedian(gtime)
            gmodflux, gmodtrend = (
                get_model_transit_quad(paramd, gmodtime, _tmid)
            )
            gmodflux -= gmodtrend

            # remove the trends before plotting
            _, gtrend = get_model_transit_quad(paramd, gtime, _tmid)
            gflux -= gtrend

        depth_TESS = np.max(gmodflux) - np.min(gmodflux)
        depth_TESS_expect = 4374e-6
        print(f'{depth_TESS_expect:.3e}, {depth_TESS:.3e}')
        if d in ['20200401', '20200426']:
            frac = DELTA_LIM_RC/depth_TESS # scale depth by this
        elif d in ['20200521']:
            frac = 1
        elif d in ['20200614']:
            frac = DELTA_LIM_B/depth_TESS

        r_TESS = np.exp(paramd['log_r'])
        r_inBp = np.sqrt(frac) * r_TESS
        log_r_inBp = np.log(r_inBp)

        bp_paramd = deepcopy(paramd)
        bp_paramd['log_r'] = log_r_inBp
        if modelid not in ['alltransit_quad', 'allindivtransit']:
            bp_modflux = (
                get_model_transit(bp_paramd, gmodtime)
            )
        else:
            bp_modflux, bp_modfluxtrend = (
                get_model_transit_quad(bp_paramd, gmodtime, _tmid)
            )
            bp_modflux -= bp_modfluxtrend

        # gtime -= t_offset
        # gmodtime -= t_offset

        bintime = 600
        bd = time_bin_magseries(gtime, gflux, binsize=bintime, minbinelems=2)
        gbintime, gbinflux = bd['binnedtimes'], bd['binnedmags']
        ##########################################


        mid_time = t0 + per*tra_ix
        tdur = 2/24. # roughly, in units of days
        n = 1.55 # sets window width
        start_time = mid_time - n*tdur
        end_time = mid_time + n*tdur

        s = (gtime > start_time) & (gtime < end_time)
        bs = (gbintime > start_time) & (gbintime < end_time)
        gs = (gmodtime > start_time) & (gmodtime < end_time)

        ax.scatter((gtime[s]-mid_time)*24,
                   (gflux[s] - np.max(gmodflux[gs]))*1e3,
                   c='darkgray', zorder=3, s=7, rasterized=False,
                   linewidths=0, alpha=0.5)

        ax.scatter((gbintime[bs]-mid_time)*24,
                   (gbinflux[bs] - np.max(gmodflux[gs]))*1e3,
                   c='black', zorder=4, s=18, rasterized=False,
                   linewidths=0)

        if modelid is None:
            l0 = (
                'TESS-only fit (' +
                f'{1e3*depth_TESS:.2f}'+'$\,$ppt)'
            )
        elif modelid in ['alltransit_quad', 'allindivtransit']:
            l0 = None
        else:
            l0 = (
                'All-transit fit (' +
                f'{1e3*depth_TESS:.2f}'+'$\,$ppt)'
            )

        if d in ['20200401', '20200426']:
            l1 = (
                '$\delta_{\mathrm{R_C}}$ > '+
                f'{1e3*DELTA_LIM_RC:.2f}'+'$\,$ppt ' +
                '(2$\sigma$)'
            )
            color = 'red'
        elif d in ['20200521']:
            l1 = None
            color = None
        elif d in ['20200614']:
            l1 = (
                '$\delta_{\mathrm{B_J}}$ > '+
                f'{1e3*DELTA_LIM_B:.2f}'+'$\,$ppt '+
                '(2$\sigma$)'
            )
            color = 'C0'

        ax.plot((gmodtime[gs]-mid_time)*24,
                (gmodflux[gs] - np.max(gmodflux[gs]))*1e3 ,
                color='gray', alpha=0.8, rasterized=False, lw=1, zorder=1,
                label=l0)

        if d in ['20200401', '20200426', '20200614']:

            ax.plot((gmodtime[gs]-mid_time)*24,
                    (bp_modflux[gs] - np.max(gmodflux[gs]))*1e3 ,
                    color=color, alpha=0.8, rasterized=False, lw=1, zorder=1,
                    label=l1)

        ax.set_ylim((-10, 6))

        props = dict(boxstyle='square', facecolor='white', alpha=0.5, pad=0.15,
                     linewidth=0)
        ax.text(np.nanpercentile(24*(gmodtime[gs]-mid_time), 97), -9.5, t,
                ha='right', va='bottom', bbox=props, zorder=-1,
                fontsize='x-small')

        if modelid in ['alltransit_quad', 'allindivtransit']:
            if d in ['20200426', '20200614']:
                props = dict(boxstyle='square', facecolor='white', alpha=0.5,
                             pad=0.15, linewidth=0)
                ax.text(np.nanpercentile(24*(gmodtime[gs]-mid_time), 3), 4.9,
                        l1, ha='left', va='top', bbox=props, zorder=-1,
                        fontsize='x-small', color=color)

        for a in [ax]:
            a.set_xlim( ( 24*(start_time-mid_time), 24*(end_time-mid_time) ) )

            ymin, ymax = a.get_ylim()
            a.vlines(
                24*(mid_time-mid_time), ymin, ymax, colors='gray', alpha=0.5,
                linestyles='--', zorder=-2, linewidths=0.5
            )
            a.set_ylim((ymin, ymax))

            if d in ['20200426', '20200614']:
                # hide the ytick labels
                labels = [item.get_text() for item in
                          a.get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                a.set_yticklabels(empty_string_labels)

            xval = np.arange(-3,4,1)
            ax.set_xticks(xval)

        if showerror:
            _e = 1e3*np.median(gflux_err)
            # ax.errorbar(
            #     -2.6, -7, yerr=_e, fmt='none',
            #     ecolor='darkgray', alpha=0.8, elinewidth=1, capsize=2,
            # )

            # bin to roughly 5e-4 * 8.3 * 24 * 60 ~= 6 minute intervals
            sampletime = np.nanmedian(np.diff(gtime))*24*60*60 # seconds
            errorfactor = (sampletime/bintime)**(1/2)

            ax.errorbar(
                -2.35, -7, yerr=errorfactor*_e,
                fmt='none', ecolor='black', alpha=1, elinewidth=1, capsize=2,
            )

            print(f'{_e:.2f}, {errorfactor*_e:.2f}')


    for ax in tra_axs:
        format_ax(ax)

    fig.text(0.5,-0.01, 'Hours from mid-transit', ha='center',
             fontsize='small')
    fig.text(-0.02,0.5, 'Relative flux [ppt]', va='center',
             rotation=90, fontsize='small')

    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_fitindiv(m, summdf, outpath, overwrite=1, modelid=None):

    if modelid != 'allindivtransit':
        raise NotImplementedError

    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return

    yval = "PDCSAP_FLUX"
    provenance = 'spoc'
    time, flux, flux_err = get_clean_tessphot(
        provenance, yval, binsize=None, maskflares=0
    )

    t_offset = np.nanmin(time)
    time -= t_offset

    FLARETIMES = [
        (4.60, 4.63),
        (37.533, 37.62)
    ]
    flaresel = np.zeros_like(time).astype(bool)
    for ft in FLARETIMES:
        flaresel |= ( (time > min(ft)) & (time < max(ft)) )

    t0 = summdf.loc['t0', 'median'] - t_offset
    per = summdf.loc['period', 'median']
    epochs = np.arange(-100,100,1)
    tra_times = t0 + per*epochs

    plt.close('all')

    ##########################################

    # figsize=(8.5, 10) full page... 10 leaves space.
    fig = plt.figure(figsize=(4, 3))

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
    yval = (flux - np.nanmedian(flux))*1e3
    ax0.scatter(time[~flaresel], yval[~flaresel], c='k', zorder=3, s=0.5,
                rasterized=False, linewidths=0)
    ax0.scatter(time[flaresel], yval[flaresel], c='darkgray', zorder=3, s=1,
                marker='x', rasterized=False, linewidth=0.1)
    ax0.set_ylim((-20, 20)) # omitting like 1 upper point from the big flare at time 38
    ymin, ymax = ax0.get_ylim()
    ax0.vlines(
        tra_times, ymin, ymax, colors='darkgray', alpha=0.5,
        linestyles='--', zorder=-2, linewidths=0.2
    )
    ax0.set_ylim((ymin, ymax))
    ax0.set_xlim((np.nanmin(time)-1, np.nanmax(time)+1))
    ax0.set_xlabel('Days from start', fontsize='small')

    # zoom-in of raw transits
    for ind, (ax, tra_ix) in enumerate(zip(tra_axs, tra_ixs)):

        mid_time = t0 + per*tra_ix
        tdur = 2/24. # roughly, in units of days
        n = 3.5
        start_time = mid_time - n*tdur
        end_time = mid_time + n*tdur

        ##########
        modtime = np.linspace(start_time, end_time, int(2e3))

        params = ['period', 't0', 'log_r', 'b', 'u[0]', 'u[1]',
                  f'tess_{ind}_mean', f'tess_{ind}_a1', f'tess_{ind}_a2',
                  'r_star', 'logg_star']

        paramd = {k:summdf.loc[k, 'median'] for k in params}

        _tmid = np.nanmedian(m.data[f'tess_{ind}'][0])

        modflux, modtrend = (
            get_model_transit_quad(paramd, modtime + t_offset, _tmid)
        )
        ##########

        s = (time > start_time) & (time < end_time)

        _flaresel = np.zeros_like(time[s]).astype(bool)
        for ft in FLARETIMES:
            _flaresel |= ( (time[s] > min(ft)) & (time[s] < max(ft)) )

        size = 6
        if np.any(_flaresel):
            ax.scatter(24*(time[s][~_flaresel]-mid_time),
                       (flux[s] - np.nanmedian(flux[s]))[~_flaresel]*1e3,
                       c='k', zorder=3, s=size, rasterized=False, linewidths=0)
            ax.scatter(24*(time[s][_flaresel]-mid_time),
                       (flux[s] - np.nanmedian(flux[s]))[_flaresel]*1e3,
                       c='darkgray', zorder=3, s=size, rasterized=False,
                       linewidth=0.1, marker='x')
        else:
            ax.scatter(24*(time[s]-mid_time),
                       (flux[s] - np.nanmedian(flux[s]))*1e3,
                       c='k', zorder=3, s=size, rasterized=False, linewidths=0)

        ax.plot(24*(modtime-mid_time), 1e3*(modflux-np.nanmedian(flux[s])),
                color='darkgray', alpha=0.8, rasterized=False, lw=1, zorder=4)

        ax.set_xlim((24*(start_time-mid_time), 24*(end_time-mid_time)))
        ax.set_ylim((-8, 8))

        # ymin, ymax = ax.get_ylim()
        # ax.vlines(
        #     24*(mid_time-mid_time), ymin, ymax, colors='darkgray', alpha=0.5,
        #     linestyles='--', zorder=-2, linewidths=0.2
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

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center',
             rotation=90, fontsize='small')
    fig.text(0.5,-0.01, 'Hours from mid-transit', ha='center',
             fontsize='small')

    fig.tight_layout(h_pad=0.5, w_pad=0.1)
    savefig(fig, outpath, writepdf=1, dpi=300)


def plot_subsetcorner(m, outpath):

    # corner plot of posterior samples
    plt.close('all')

    varnames = ['log_r', 'b', 'rho_star']
    labels = ['$\log(R_\mathrm{p}/R_{\star})$', '$b$',
              r'$\rho_\star$ [$\mathrm{g}\,\mathrm{cm}^{-3}$]']

    trace_df = pm.trace_to_dataframe(m.trace, varnames=varnames)

    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
                        show_titles=False, title_kwargs={"fontsize": 12},
                        title_fmt='.2g', labels=labels,
                        label_kwargs={"fontsize":16})

    axs = fig.axes

    for a in axs:
        format_ax(a)
    # fig.tight_layout()

    savefig(fig, outpath, writepdf=0, dpi=300)
