"""
TESS image (x's numbered)
DSS2 Red image.
"""

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from timmy.plotting import plot_scene

def main():
    # J2015.5 gaia
    ra = 157.03728055645
    dec = -64.50521068147
    ticid = '460205581'

    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    # NOTE: S11 vs S10: not particularly different

    # hdul = fits.open(
    #     '/Users/luke/Dropbox/proj/timmy/data/MAST_2020-05-04T1852/TESS/tess2019112060037-s0011-0000000460205581-0143-s/'
    #     'tess2019112060037-s0011-0000000460205581-0143-s_tp.fits'
    # )

    hdul = fits.open(
        '/Users/luke/Dropbox/proj/timmy/data/MAST_2020-05-04T1852/TESS/tess2019085135100-s0010-0000000460205581-0140-s/'
        'tess2019085135100-s0010-0000000460205581-0140-s_tp.fits'
    )

    img_wcs = wcs.WCS(hdul[2].header)

    d = hdul[1].data
    mean_img = np.nansum(d["FLUX"], axis=0) / d["FLUX"].shape[0]

    # 37, 43 for output A (PTFO)
    # 133, 139 for output C (837)
    bkgd_mask = (hdul[2].data == 133)
    ap_mask = (hdul[2].data == 139)

    outpath = '../results/paper_plots/scene.png'
    plot_scene(c_obj, img_wcs, mean_img, outpath, ap_mask=ap_mask,
               bkgd_mask=bkgd_mask, ticid=ticid, Tmag_cutoff=16,
               showdss=0)


if __name__ == "__main__":
    main()
