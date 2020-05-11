"""
P Evans acquired 2 runs on the El Sauce 36cm.
Are they sufficient to show that TOI 837 is truly on target?
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
import os

from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

import timmy.plotting as tp
from timmy.paths import DATADIR, PHOTDIR, RESULTSDIR

def main():

    vis_groundimgs('2020-04-01')
    vis_groundimgs('2020-04-26')


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


def vis_groundimgs(datestr='2020-04-01'):

    imgpaths, outdir = _init_dir(datestr)

    # J2015.5 gaia
    ra = 157.03728055645
    dec = -64.50521068147
    ticid = '460205581'
    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    xmin, xmax = 710, 790
    ymin, ymax = 482, 562

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


if __name__ == "__main__":
    main()
