"""
P Evans acquired 2 runs on the El Sauce 36cm.
Are they sufficient to show that TOI 837 is truly on target?
"""

import numpy as np
from astropy import units as u
import timmy.groundphot as tgp
import timmy.plotting as tp

def main():

    do_vis_groundimgs = 0
    do_shift_groundimgs = 0
    do_pixel_lc = 0
    do_pu_apphot = 1

    datestrs = ['2020-04-01', '2020-04-26']

    for datestr in datestrs:

        if do_vis_groundimgs:
            tgp.vis_groundimgs(datestr)

        if do_shift_groundimgs:
            tgp.shift_groundimgs(datestr)

        if do_pixel_lc:
            tgp.pixel_lc(datestr)

        if do_pu_apphot:

            tgp.photutils_apphot(datestr)
            tgp.format_photutils_lcs(datestr)

            for apn in range(0,7):
                ap = 'aperture_sum_{}'.format(apn)
                tp.vis_photutils_lcs(datestr, ap)
                tgp.compstar_detrend(datestr, ap, target='837')

            # detrend custom aperture LCs (r=2px).
            npxshift = -np.arange(0,6.5,0.5)
            px_scale = 0.734*u.arcsec # per pixel
            for ix, npx in enumerate(npxshift):
                # the custom aperture lightcurves are 1-based
                ap_ind = ix+1
                ap = 'aperture_sum_{}'.format(ap_ind)
                tgp.compstar_detrend(datestr, ap, target='customap')


if __name__ == "__main__":
    main()
