import matplotlib.pyplot as plt, numpy as np
import scipy.misc as msc
import timmy.imgproc as ti
from astropy.io import fits
import astropy.visualization as vis

img0 = msc.face()[:,:,0] # rgb image, take one channel
hl = fits.open('/Users/luke/Dropbox/proj/timmy/data/phot/2020-04-01/TIC460205581-01-0196_Rc1_out.fit')
img1 = hl[0].data
hl.close()

for ix, img in enumerate([img0,img1]):

    if ix == 1:
        vmin,vmax = 10, int(1e4)
        norm = vis.ImageNormalize(
            vmin=vmin, vmax=vmax, stretch=vis.LogStretch(1000))
    else:
        norm = None

    f, axs = plt.subplots(nrows=2,ncols=2)
    # note: this image really should have origin='upper' (otherwise trashpanda is upside-down)
    # but this is to match fits image processing convention
    axs[0,0].imshow(img, cmap=plt.cm.gray, origin='lower', norm=norm)
    axs[0,0].set_title('shape: {}'.format(img.shape))

    dx,dy = 200,50
    axs[1,0].imshow(ti.integer_shift_img(img, dx, dy), cmap=plt.cm.gray, origin='lower', norm=norm)
    axs[1,0].set_title('dx={}, dy={}'.format(dx,dy))

    dx,dy = -200,50
    axs[0,1].imshow(ti.integer_shift_img(img, dx, dy), cmap=plt.cm.gray, origin='lower', norm=norm)
    axs[0,1].set_title('dx={}, dy={}'.format(dx,dy))

    dx,dy = -200,-50
    axs[1,1].imshow(ti.integer_shift_img(img, dx, dy), cmap=plt.cm.gray, origin='lower', norm=norm)
    axs[1,1].set_title('dx={}, dy={}'.format(dx,dy))

    f.tight_layout()
    f.savefig('../results/test_results/test_shift_{}.png'.format(ix), dpi=250)
