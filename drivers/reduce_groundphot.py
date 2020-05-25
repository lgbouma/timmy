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
    do_stackviz_blendcheck = 1
    do_vis_groundimg_customap = 0

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

            # detrend TOI 837 relative to comparison stars
            for apn in range(0,7):
                ap = 'aperture_sum_{}'.format(apn)
                tp.vis_photutils_lcs(datestr, ap)
                tgp.compstar_detrend(datestr, ap, target='837')

            # detrend custom aperture LCs relative to comparison stars
            npxshift = -np.arange(0,6.5,0.5)
            px_scale = 0.734*u.arcsec # per pixel
            for ix, npx in enumerate(npxshift):
                for apn in range(0,7):
                    ap = 'aperture_sum_{}'.format(apn)
                    _id = str(ix+1)
                    tgp.compstar_detrend(datestr, ap, target='customap',
                                         customid=_id)

        if do_stackviz_blendcheck:
            for apn in range(0,7):
                tp.stackviz_blend_check(datestr, apn, soln=0, adaptiveoffset=1)
                tp.stackviz_blend_check(datestr, apn, soln=0, adaptiveoffset=0)

        if do_vis_groundimg_customap:
            tgp.vis_groundimgs(datestr, customap=1)

if __name__ == "__main__":
    main()
