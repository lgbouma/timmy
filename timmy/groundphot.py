import numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
import os
from copy import deepcopy
from numpy import array as nparr

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import hstack, Table
from astropy.wcs import WCS

import photutils.aperture as pa
from astroquery.mast import Catalogs

from sklearn.linear_model import LinearRegression

import timmy.plotting as tp
from timmy.paths import DATADIR, PHOTDIR, RESULTSDIR
import timmy.imgproc as ti

def _init_dir(datestr):

    imgdir = os.path.join(PHOTDIR, datestr)
    imgpaths = glob(os.path.join(imgdir, '*.fit'))
    outdir = os.path.join(RESULTSDIR, 'groundphot', datestr)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'vis_groundimgs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    return imgpaths, outdir


def __init_dir(datestr):

    imgdir = os.path.join(PHOTDIR, datestr)
    imgpaths = glob(os.path.join(imgdir, '*.fit'))
    outdir = os.path.join(RESULTSDIR, 'groundphot', datestr)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'shift_groundimgs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    return imgpaths, outdir


def ___init_dir(datestr):

    # shifted images
    imgdir = '/Users/luke/Dropbox/proj/timmy/results/groundphot/{}/shift_groundimgs'.format(datestr)
    imgpaths = glob(os.path.join(imgdir, '*.fits'))

    outdir = os.path.join(RESULTSDIR, 'groundphot', datestr)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'pixel_lc')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    return imgpaths, outdir


def ____init_dir(datestr):

    imgdir = os.path.join(PHOTDIR, datestr)
    imgpaths = glob(os.path.join(imgdir, '*.fit'))
    outdir = os.path.join(RESULTSDIR, 'groundphot', datestr)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'photutils_apphot')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    return imgpaths, outdir



def vis_groundimgs(datestr='2020-04-01', customap=0):

    imgpaths, outdir = _init_dir(datestr)

    # J2015.5 gaia
    ra = 157.03728055645
    dec = -64.50521068147
    ticid = '460205581'
    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    xmin, xmax = 710, 790  # note: xmin/xmax in mpl coordinates (not ndarray coordinates)
    ymin, ymax = 482, 562 # note: ymin/ymax in mpl coordinates (not ndarray coordinates)
    if customap:
        xmin, xmax = 710+12, 770+12  # note: xmin/xmax in mpl coordinates (not ndarray coordinates)
        ymin, ymax = 492, 552 # note: ymin/ymax in mpl coordinates (not ndarray coordinates)

    if customap:
        # only do one image for customap showing
        imgpath = imgpaths[100]
        outpath = os.path.join(
            outdir, os.path.basename(imgpath).replace('.fit','_groundscene.png')
        )

        hdul = fits.open(imgpath)
        img = hdul[0].data
        img_wcs = wcs.WCS(hdul[0].header)

        xlim = (xmin, xmax)
        ylim = (ymin, ymax)

        tp.plot_groundscene(c_obj, img_wcs, img, outpath, Tmag_cutoff=16,
                            showcolorbar=0, ticid=ticid, xlim=xlim,
                            ylim=ylim, customap=customap)


    else:
        for ix, imgpath in enumerate(imgpaths):
            outpath = os.path.join(
                outdir, os.path.basename(imgpath).replace('.fit','_groundscene.png')
            )
            if not os.path.exists(outpath):

                hdul = fits.open(imgpath)
                img = hdul[0].data
                img_wcs = wcs.WCS(hdul[0].header)

                xlim = (xmin, xmax)
                ylim = (ymin, ymax)

                tp.plot_groundscene(c_obj, img_wcs, img, outpath, Tmag_cutoff=16,
                                    showcolorbar=1, ticid=ticid, xlim=xlim,
                                    ylim=ylim, customap=customap)
            else:
                print('found {}'.format(outpath))



def shift_groundimgs(datestr):
    """
    shift target star to the image center
    """

    imgpaths, outdir = __init_dir(datestr)

    # J2015.5 gaia
    ra, dec = 157.03728055645, -64.50521068147
    ticid = '460205581'
    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    x0, y0 = 768, 512  # in array coordinates
    xmin, xmax = x0-50, x0+50  # note: xmin/xmax in mpl coordinates (not ndarray coordinates)
    ymin, ymax = y0-50, y0+50 # note: ymin/ymax in mpl coordinates (not ndarray coordinates)

    for imgpath in imgpaths:
        outpath = os.path.join(
            outdir, os.path.basename(imgpath).replace('.fit','_shift.png')
        )
        if not os.path.exists(outpath):

            hdul = fits.open(imgpath)
            img = hdul[0].data
            hdr = hdul[0].header
            hdul.close()
            img_wcs = wcs.WCS(hdr)

            _ra, _dec = float(c_obj.ra.value), float(c_obj.dec.value)
            target_x, target_y = img_wcs.all_world2pix(_ra,_dec,0)

            dx = int(x0 - target_x)
            dy = int(y0 - target_y)

            shift_img = ti.integer_shift_img(img, dx, dy)

            titlestr0 = os.path.basename(imgpath)
            titlestr1 = 'shift dx={}, dy={}'.format(dx,dy)

            xlim = (xmin, xmax)
            ylim = (ymin, ymax)

            tp.shift_img_plot(img, shift_img, xlim, ylim, outpath, x0, y0,
                              target_x, target_y, titlestr0, titlestr1,
                              showcolorbar=0)

            hdr['SHIFTX'] = dx
            hdr['SHIFTY'] = dy
            outfits = outpath.replace('.png', '.fits')

            if not os.path.exists(outfits):
                outhdu = fits.PrimaryHDU(data=shift_img, header=hdr)
                outhdul = fits.HDUList([outhdu])
                outhdul.writeto(outfits)
                print('made {}'.format(outfits))
                outhdul.close()
            else:
                print('found {}'.format(outfits))

        else:
            print('found {}'.format(outpath))


def get_image_cube(imgpaths):

    imgs = []
    hdrs = []
    times = []
    for imgpath in imgpaths:
        hl = fits.open(imgpath)
        img = hl[0].data
        hdr = hl[0].header
        imgs.append(img)
        hdrs.append(hdr)
        times.append(hdr['BJD_TDB'])
        hl.close()

    # shape: (408, 1024, 1536). so N_time x N_y x N_x
    img_cube = np.array(imgs)
    times = np.array(times)

    return img_cube, hdrs, times


def pixel_lc(datestr):
    # NOTE: lightkurve .interact() might be worth hacking for this

    # get shifted image paths
    imgpaths, outdir = ___init_dir(datestr)

    img_cube, hdrs, times = get_image_cube(imgpaths)

    outpath = os.path.join(outdir, 'pixel_lc.png')

    showvlines = 1 if datestr=='2020-04-01' else 0
    tp.plot_pixel_lc(times, img_cube, outpath, showvlines=showvlines)


def get_nbhr_stars():

    ra, dec = 157.03728055645, -64.50521068147
    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    #
    # get the neighbor stars, and their positions
    #
    Tmag_cutoff = 16
    radius = 6.0*u.arcminute

    nbhr_stars = Catalogs.query_region(
        "{} {}".format(float(c_obj.ra.value), float(c_obj.dec.value)),
        catalog="TIC",
        radius=radius
    )

    sel = (nbhr_stars['Tmag'] < Tmag_cutoff)

    sra, sdec, ticids = (
        nbhr_stars[sel]['ra'],
        nbhr_stars[sel]['dec'],
        nbhr_stars[sel]['ID']
    )

    return nbhr_stars, sra, sdec, ticids


def photutils_apphot(datestr):
    """
    Run aperture photometry using photutils.
    Extract LCs for Tmag < 16 stars within 6 arcminutes of TOI 837.
    And also do the "custom apertures" along a line between TOI 837 and Star A.
    """

    imgpaths, outdir = ____init_dir(datestr)

    # J2015.5 gaia TOI 837
    ra, dec = 157.03728055645, -64.50521068147
    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    # J2015.5 gaia, Star A = TIC 847769574 (T=14.6). $2.3$'' west
    # == Gaia 5251470948222139904
    A_ra, A_dec = 157.03581886712300, -64.50508245570860
    c_StarA = SkyCoord(A_ra, A_dec, unit=(u.deg), frame='icrs')

    # will do photometry along the line separating these two stars.
    posn_angle = c_obj.position_angle(c_StarA)
    sep = c_obj.separation(c_StarA)

    # number of pixels to shift. sign was checked empirically.
    npxshift = -np.arange(0,6.5,0.5)
    px_scale = 0.734*u.arcsec # per pixel
    line_ap_locs = []
    for n in npxshift:
        line_ap_locs.append(
            c_StarA.directional_offset_by(posn_angle, n*px_scale)
        )

    # sanity check: 3 pixel shift should be roughly location of TOI 837
    assert np.round(ra,3) == np.round(line_ap_locs[6].ra.value, 3)

    nbhr_stars, sra, sdec, ticids = get_nbhr_stars()
    positions = SkyCoord(sra, sdec, unit=(u.deg), frame='icrs')

    #
    # apertures of radii 1-7 pixels, for all stars in image
    #
    n_pxs = range(1,8)

    aplist = []
    for n in n_pxs:
        aplist.append(pa.SkyCircularAperture(positions, r=n*px_scale))

    #
    # finally, do the photometry.
    #
    for imgpath in imgpaths:

        outpath = os.path.join(
            outdir, os.path.basename(imgpath).replace(
                '.fit', '_phottable.fits')
        )

        if os.path.exists(outpath):
            print('found {}, skip.'.format(outpath))
            continue

        hdul = fits.open(imgpath)
        img = hdul[0].data
        img_wcs = wcs.WCS(hdul[0].header)
        hdul.close()

        phot_table = pa.aperture_photometry(img, aplist, wcs=img_wcs)

        #
        # apertures of radii 1-7 pixels, along the line between Star A and TOI
        # 837.
        #
        custom_aplist = []
        custom_phottables = []

        line_ap_locs = SkyCoord(line_ap_locs)
        for n in range(1,8):
            custom_aplist.append(pa.SkyCircularAperture(line_ap_locs, r=n*px_scale))

        custom_phot_table = pa.aperture_photometry(
            img, custom_aplist, wcs=img_wcs
        )

        #
        # save both
        #
        outpath = os.path.join(
            outdir, os.path.basename(imgpath).replace(
                '.fit', '_phottable.fits')
        )
        phot_table.write(outpath, format='fits')
        print('made {}'.format(outpath))

        outpath = os.path.join(
            outdir, os.path.basename(imgpath).replace(
                '.fit', '_customtable.fits')
        )
        custom_phot_table.write(outpath, format='fits')
        print('made {}'.format(outpath))


def format_photutils_lcs(datestr):

    outdir = os.path.join(RESULTSDIR, 'groundphot', datestr,
                          'vis_photutils_lcs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if len(glob(os.path.join(outdir,'*csv'))) > 0:
        print('Found LCs already made. Skip')
        return

    _, photdir = ____init_dir(datestr)

    phottable_paths = np.sort(glob(os.path.join(photdir, '*phottable.fits')))
    assert len(phottable_paths) > 0

    customtable_paths = np.sort(glob(os.path.join(photdir, '*customtable.fits')))
    assert len(customtable_paths) > 0

    cal_paths = np.sort(glob(os.path.join(DATADIR, 'phot', datestr, '*.fit')))
    assert len(cal_paths) == len(phottable_paths)

    times, airmasses = [], []
    data = []
    cdata = []

    for pt, ct, cp in zip(phottable_paths, customtable_paths, cal_paths):

        with fits.open(pt) as hdul:
            d = deepcopy(hdul[1].data)
            data.append(d)
            del hdul[1].data

        with fits.open(ct) as hdul:
            cd = deepcopy(hdul[1].data)
            cdata.append(cd)
            del hdul[1].data

        with fits.open(cp) as hdul:
            time = deepcopy(hdul[0].header['BJD_TDB'])
            times.append(time)
            airmass = deepcopy(hdul[0].header['AIRMASS'])
            airmasses.append(airmass)
            del hdul[0].header['BJD_TDB']
            del hdul[0].header['AIRMASS']

    data = np.array(data)
    cdata = np.array(cdata)

    assert np.count_nonzero(np.diff(data['sky_center.ra'], axis=0)) == 0
    assert np.count_nonzero(np.diff(data['sky_center.dec'], axis=0)) == 0

    # transpose: flux measurements at distinct times -> flux vs time per
    # source.
    # this is significantly easier because it's not lossy transposition.
    # we know the locations, and ticids, of all the sources. and they're
    # already arranged in correct identifier order!
    # the assert statements here ensure this is true.

    nbhr_stars, sra, sdec, ticids = get_nbhr_stars()

    assert np.count_nonzero(np.array(data['sky_center.ra'][0]) - sra) == 0

    tdata = data.T

    for lc, ticid in zip(tdata, ticids):

        outpath = os.path.join(
            outdir, 'TIC'+str(ticid).zfill(14)+'_photutils_groundlc.csv'
        )

        outdf = pd.DataFrame(lc)

        outdf['ticid'] = ticid
        outdf['BJD_TDB'] = times
        outdf['airmass'] = airmasses

        outdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))


    tcdata = cdata.T

    for lc in tcdata:

        _id = str(lc['id'][0])

        outpath = os.path.join(
            outdir, 'CUSTOM'+str(_id).zfill(4)+'_photutils_groundlc.csv'
        )

        outdf = pd.DataFrame(lc)

        outdf['BJD_TDB'] = times
        outdf['airmass'] = airmasses

        outdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

    print('Finished transposing/formatting LCs.')



# key: aperture number + observation date
# values: list of bad comparison stars
BADCOMPSTARS = {
    '0_2020-04-01': ["847769574"],
    '1_2020-04-01': ["847769574"],
    '2_2020-04-01': [],
    '3_2020-04-01': [],
    '4_2020-04-01': [],
    '5_2020-04-01': ["460205587"],
    '6_2020-04-01': ["847770388"],
    '0_2020-04-26': ["847769574"],
    '1_2020-04-26': ["847769574"],
    '2_2020-04-26': [],
    '3_2020-04-26': [],
    '4_2020-04-26': [],
    '5_2020-04-26': ["460205587"],
    '6_2020-04-26': ["847770388"],
    '0_2020-05-21': ["847769574"],
    '1_2020-05-21': ["847769574"],
    '2_2020-05-21': [],
    '3_2020-05-21': [],
    '4_2020-05-21': [],
    '5_2020-05-21': ["460205587"],
    '6_2020-05-21': ["847770388"],

}

def compstar_detrend(datestr, ap, target='837', customid=None):
    """
    target_lc = Σ c_i * f_i, for f_i comparison lightcurves. solve for the c_i
    via least squares.

    kwargs:
        target (str): '837' or 'customap'; '837' means correctly centered
        apertures on TOI 837; 'customap' means the weird "along line" apertures
        between Star A and TOI 837.
    """

    if target=='customap':
        assert isinstance(customid, str)

    if datestr == '2020-04-01':
        N_drop = 47 # per Phil Evan's reduction notes
    elif datestr == '2020-04-26':
        N_drop = 0
    elif datestr == '2020-05-21':
        N_drop = 0
    else:
        raise NotImplementedError('pls manually set N_drop')

    #
    # get target star flux
    #
    if target=='837':
        ticid = '460205581'
        targetpath = glob(os.path.join(
            RESULTSDIR,  'groundphot', datestr, 'vis_photutils_lcs',
            'TIC*{}*_photutils_groundlc.csv'.format(ticid)
        ))[0]
        targetdf = pd.read_csv(targetpath)
    elif target=='customap':
        targetpath = os.path.join(
            RESULTSDIR,  'groundphot', datestr, 'vis_photutils_lcs',
            'CUSTOM'+str(customid).zfill(4)+'_photutils_groundlc.csv'
        )
        targetdf = pd.read_csv(targetpath)
    else:
        raise NotImplementedError

    time = nparr(targetdf['BJD_TDB'])[N_drop:]
    target_flux = nparr(targetdf[ap])[N_drop:]
    target_flux /= np.nanmean(target_flux)

    #
    # get comparison star fluxes
    #
    incsvpath = os.path.join(
        RESULTSDIR,  'groundphot', datestr, 'vis_photutils_lcs',
        'vis_photutils_lcs_compstars_{}.csv'.format(ap)
    )

    comp_df = pd.read_csv(incsvpath)

    comp_ticids = nparr([c.split('_')[-1] for c in list(comp_df.columns) if
                         c.startswith('time_')])

    bad_ticids = nparr(BADCOMPSTARS[ap[-1]+'_'+datestr])

    if len(bad_ticids) > 0:

        print('not using as comparison stars {}'.format(bad_ticids))

        comp_ticids = np.setdiff1d(
            comp_ticids, bad_ticids
        )

    comp_fluxs = nparr(
        [nparr(comp_df['flux_{}'.format(t)]) for t in comp_ticids]
    )

    #
    # regress
    #

    if np.any(pd.isnull(target_flux)):
        raise NotImplementedError

    reg = LinearRegression(fit_intercept=True)

    mean_flux = np.nanmean(target_flux)

    _X = comp_fluxs[:, :]

    reg.fit(_X.T, target_flux-mean_flux)

    model_flux = reg.intercept_ + (reg.coef_ @ _X)

    model_flux += mean_flux

    # divide, not subtract, b/c flux, not mag.
    flat_flux = target_flux / model_flux

    outdir = os.path.join(
        RESULTSDIR,  'groundphot', datestr, 'compstar_detrend'
    )
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if target=='837':
        outpath = os.path.join(outdir, 'toi837_detrended_{}.png'.format(ap))
    elif target=='customap':
        outpath = os.path.join(
            outdir,
            'toi837_detrended_customap_{}_{}.png'.
            format(ap, str(customid).zfill(4))
        )

    provenance = 'Evans_{}'.format(datestr)
    if target=='837':
        titlestr = 'Evans {}'.format(datestr)
    elif target=='customap':
        titlestr = 'Evans {}. posn {}. ap {}'.format(datestr, customid, ap)

    tp._plot_quicklooklc(
        outpath, time, target_flux, target_flux*1e-3, flat_flux, model_flux,
        showvlines=1, figsize=(18,8), provenance=provenance, timepad=0.05,
        titlestr=titlestr, ylim=(0.985, 1.015)
    )

    # save the LC
    if target=='837':
        outpath = os.path.join(outdir, 'toi837_detrended_{}.csv'.format(ap))
    elif target=='customap':
        outpath = os.path.join(
            outdir,
            'toi837_detrended_customap_{}_{}.csv'.
            format(ap, str(customid).zfill(4))
        )

    outdf = pd.DataFrame({})
    outdf['time'] = time
    outdf['flux'] = target_flux
    outdf['flux_err'] = target_flux*1e-3 # hacky, but fine
    outdf['flat_flux'] = flat_flux
    outdf['model_flux'] = model_flux

    outdf.to_csv(outpath, index=False)
    print('wrote {}'.format(outpath))
