"""
P Evans acquired 2 runs on the El Sauce 36cm.
Are they sufficient to show that TOI 837 is truly on target?
"""

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

            #FIXME: implement!
            tp.vis_photutils_lcs(datestr)


if __name__ == "__main__":
    main()
