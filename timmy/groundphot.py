import numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
import os
from copy import deepcopy

from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import hstack, Table

import photutils.aperture as pa
from astropy.wcs import WCS
from astroquery.mast import Catalogs

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



def vis_groundimgs(datestr='2020-04-01'):

    imgpaths, outdir = _init_dir(datestr)

    # J2015.5 gaia
    ra = 157.03728055645
    dec = -64.50521068147
    ticid = '460205581'
    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    xmin, xmax = 710, 790  # note: xmin/xmax in mpl coordinates (not ndarray coordinates)
    ymin, ymax = 482, 562 # note: ymin/ymax in mpl coordinates (not ndarray coordinates)

    for imgpath in imgpaths:
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
                                ylim=ylim)
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
        # apertures of radius 2 pixels, along the line between Star A and TOI
        # 837.
        #
        custom_aplist = []
        custom_phottables = []

        for loc in line_ap_locs:

            aperture = pa.SkyCircularAperture(loc, r=2*px_scale)

            custom_phot_table = pa.aperture_photometry(
                img, aperture, wcs=img_wcs
            )

            custom_phottables.append(custom_phot_table)

        custom_stack = hstack(custom_phottables)

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
        custom_stack.write(outpath, format='fits')
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

    outpath = os.path.join(
        outdir, 'custom_toi837_apertures_photutils_groundlc.csv'
    )

    custom_df = Table(cdata[:,0]).to_pandas()
    custom_df['BJD_TDB'] = times
    custom_df['airmass'] = airmasses

    custom_df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
