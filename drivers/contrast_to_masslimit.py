"""
We would like to convert the contrast ratios obtained through the SOAR imaging
to constraints on the masses of putative companions, and their projected
separations from the host star.

To do this, we followed the methodology of Montet+2014, and opted to employ the
Baraffe+2003 models for substellar mass objects and the MESA isochrones for
stellar mass objects (Choi et al XXXX).

We assumed that the system age was 35 Myr, to ensure that companions would be
at a plausible state of contraction.

Due to the specific filters of the SOAR HRcam imager, we further assumed that all
sources had blackbody spectra. While this is clearly false, we do not readily
have access to the planetary and stellar atmosphere models needed for the
correct calculation. However, the Baraffe+2003 and MESA models do report
effective temperatures and bolometric luminosities.  We opt to use these
theoretical quantities, combined with the measured transmission functions, to
calculate the absolute magnitudes in the HRCam bandpasses.  The apparent
magnitudes follow from information about the distance of the star.

Applying the same assumption to TOI-837 itself, we compute its apparent
magnitude in each of the Zorro filters, and finally derive the transformation
from contrast to companion mass.
"""

##########
# config #
##########

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from numpy import trapz
from scipy.interpolate import interp1d

from timmy.read_mist_model import ISO

from astropy import units as u
from astropy import constants as const

from cdips.plotting import savefig

datadir = '../data/companion_isochrones/'

def get_merged_companion_isochrone(age=5e9, mstar=1):

    outpath = f'../data/companion_isochrones/MIST_plus_Baraffe_merged_{age:.1e}.csv'

    if not os.path.exists(outpath):

        #
        # Get Teff, L for the stellar models of MIST at 5 Gyr.
        #

        if age == 5e9:
            # MIST isochrones, v1.2 from the website
            mistpath = os.path.join(
                datadir, 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso'
            )
            iso = ISO(mistpath)

            # 10**9.7 = 5.01 Gyr. I'm OK with not interpolating further here.
            mist_age_ind = iso.age_index(9.7)
            mist_logTeff = iso.isos[mist_age_ind]['log_Teff']
            mist_logL = iso.isos[mist_age_ind]['log_L']
            mist_initial_mass = iso.isos[mist_age_ind]['initial_mass']

        elif age == 3.5e7:
            # MIST isochrones, v1.2, interpolated on website
            mistpath = os.path.join(
                datadir, 'MIST_iso_5ecfe9b1811ee.iso'
            )
            iso = ISO(mistpath)

            assert len(iso.isos) == 1

            mist_logTeff = iso.isos[0]['log_Teff']
            mist_logL = iso.isos[0]['log_L']
            mist_initial_mass = iso.isos[0]['initial_mass']


        mist_df = pd.DataFrame({
            'mass': mist_initial_mass,
            'lum': 10**(mist_logL),
            'teff': 10**(mist_logTeff),
        })

        #
        # There is one point that overlaps: Mstar = 0.1Msun. For that point,
        # use the Baraffe model. 
        #
        sel = (mist_df.mass < mstar) & (mist_df.mass > 0.1)
        mist_df = mist_df[sel]

        #
        # Get Teff, L for the Baraffe03 models 5 Gyr.
        #

        if age==5e9:
            agestr = '5gyr'
        elif age==3.5e7:
            agestr = '50myr'
        else:
            raise NotImplementedError

        # Baraffe+2003 isochrones for substellar mass objects
        bar_df = pd.read_csv(os.path.join(datadir, f'COND03_{agestr}.csv'),
                             delim_whitespace=True)

        bar_df = bar_df.drop(
            ['g','R','Mv','Mr','Mi','Mj','Mh','Mk','Mll','Mm'], axis=1
        )

        bar_df['L/Ls'] = 10**nparr(bar_df['L/Ls'])

        bar_df = bar_df.rename(
            columns={'L/Ls':'lum', 'Teff':'teff', 'M/Ms': 'mass'}
        )

        #
        # merge
        #
        mdf = pd.concat((bar_df, mist_df), sort=False).reset_index()

        mdf = mdf.drop(['index'], axis=1)

        mdf.to_csv(outpath, index=False)

    return pd.read_csv(outpath)


def abs_mag_in_bandpass(lum, teff, bandpass='562'):
    """
    lum: bolometric luminosity in units of Lsun
    teff: effective temperature in units of K
    bandpass: '562' or '832', nanometers.
    """

    if bandpass not in ['562','832','NIRC2_Kp','HRcam_Ic']:
        raise ValueError

    if bandpass in ['562', '832']:

        bandpassdir = '../data/WASP4_zorro_speckle/filters/'
        bandpasspath = os.path.join(
            bandpassdir, 'filter_EO_{}.csv'.format(bandpass)
        )

        bpdf = pd.read_csv(bandpasspath, delim_whitespace=True)

        # the actual tabulated values here are bogus at the long wavelength end.
        # obvious from like... physics, assuming detectors are silicon. (confirmed
        # by Howell in priv. comm.) 

        width = 100 # nanometers, around the bandpass middle
        sel = np.abs(float(bandpass) - bpdf.nm) < width

        bpdf = bpdf[sel]

    elif bandpass == 'NIRC2_Kp':

        # NIRC2 Kp band filter from 
        # http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=Keck/NIRC2.Kp
        bandpassdir = '../data/WASP4_NIRC2/'
        bandpasspath = os.path.join(
            bandpassdir, 'Keck_NIRC2.Kp.dat'
        )

        bpdf = pd.read_csv(bandpasspath, delim_whitespace=True,
                           names=['wvlen_angst', 'Transmission'])

        bpdf['nm'] = bpdf.wvlen_angst / 10

    elif bandpass == 'HRcam_Ic':

        # HRcam Ic band from Tokovinin+2018, with webplotdigitzer
        bandpassdir = '../data/speckle/'
        bandpasspath = os.path.join(
            bandpassdir, 'filter_HRcam_Ic.csv'
        )

        bpdf = pd.read_csv(bandpasspath)

        # percent to fraction
        bpdf['Transmission'] /= 100


    else:
        raise NotImplementedError

    #
    # see /doc/20200121_blackbody_mag_derivn.pdf for relevant discussion of
    # units and where the equations come from.
    # (for WASP-4. nothing changes for other stars)
    #

    from astropy.modeling.models import BlackBody

    M_Xs = []
    for temperature, luminosity in zip(teff*u.K, lum*u.Lsun):

        bb = BlackBody(temperature=temperature)

        wvlen = nparr(bpdf.nm)*u.nm
        B_nu_vals = bb(wvlen)
        B_lambda_vals = B_nu_vals * (const.c / wvlen**2)

        T_lambda = nparr(bpdf.Transmission)

        F_X = 4*np.pi*u.sr * trapz(B_lambda_vals * T_lambda, wvlen)

        F = const.sigma_sb * temperature**4

        # https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
        M_bol_sun = 4.83
        M_bol_star = (
            -5/2 * np.log10(luminosity/(1*u.Lsun)) + M_bol_sun
        )

        # bolometric magnitude of the star, in the bandpass!
        M_X = M_bol_star - 5/2*np.log10( F_X/F )

        M_Xs.append(M_X.value)

    return nparr(M_Xs)


def get_wasp4_mag_to_companion_contrasts(age=5e9):

    outpath = (
        '../data/WASP4_high_resoln_imaging/wasp4_mag_to_companion_contrasts.csv'
    )

    if not os.path.exists(outpath):

        # WASP-4 params
        teff = 5400 # K
        rstar = 0.893
        lum = 4*np.pi*(rstar*u.Rsun)**2 * const.sigma_sb*(teff*u.K)**4
        w4_dict = {
            'mass': 0.864,
            'teff': teff,
            'lum': lum.to(u.Lsun).value
        }

        df = get_merged_companion_isochrone(age=age, mstar=mstar)

        for bp in ['562', '832', 'NIRC2_Kp']:
            df['M_{}'.format(bp)] = abs_mag_in_bandpass(
                nparr(df.lum), nparr(df.teff), bp)

            M_bp_wasp4 = float(abs_mag_in_bandpass(
            [w4_dict['lum']], [w4_dict['teff']], bp))

            df['dmag_{}'.format(bp)] = df['M_{}'.format(bp)] - M_bp_wasp4

        df.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

    return pd.read_csv(outpath)


def get_toi837_mag_to_companion_contrasts(age=3.5e7):

    outpath = (
        '../data/speckle/toi837_mag_to_companion_contrasts.csv'
    )

    if not os.path.exists(outpath):

        from timmy.priors import TEFF, RSTAR, MSTAR

        # TOI837 params
        teff = TEFF
        rstar = RSTAR
        mstar = MSTAR
        lum = 4*np.pi*(rstar*u.Rsun)**2 * const.sigma_sb*(teff*u.K)**4
        star_dict = {
            'mass': mstar,
            'teff': teff,
            'lum': lum.to(u.Lsun).value
        }

        df = get_merged_companion_isochrone(age, mstar=mstar)

        for bp in ['HRcam_Ic']:
            df['M_{}'.format(bp)] = abs_mag_in_bandpass(
                nparr(df.lum), nparr(df.teff), bp)

            M_bp_star = float(abs_mag_in_bandpass(
            [star_dict['lum']], [star_dict['teff']], bp))

            df['dmag_{}'.format(bp)] = df['M_{}'.format(bp)] - M_bp_star

        df.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

    return pd.read_csv(outpath)



def get_companion_bounds(instrument):

    if instrument == 'Zorro':

        zorrostr = 'WASP-4_20190928_832'
        outpath = (
            '../data/WASP4_zorro_speckle/{}_companionbounds.csv'.
            format(zorrostr)
        )

        if not os.path.exists(outpath):

            df = get_wasp4_mag_to_companion_contrasts()

            #
            # WASP-4_20190928_832.dat is the most contstraining curve for basically any
            # substellar mass companion. The blackbody curve works against us in 562,
            # and the seeing was better on 20190928 than 20190912.
            #
            datapath = '../data/WASP4_zorro_speckle/{}.dat'.format(zorrostr)
            zorro_df = pd.read_csv(
                datapath, comment='#', skiprows=29,
                names=['ang_sep', 'delta_mag'], delim_whitespace=True
            )

            #
            # Interpolation function to convert observed deltamags to deltamass.
            #
            fn_dmag_to_mass = interp1d(
                nparr(df.dmag_832),
                nparr(df.mass),
                kind='quadratic',
                bounds_error=False,
                fill_value=np.nan
            )

            zorro_df['m_comp/m_sun'] = fn_dmag_to_mass(zorro_df.delta_mag)

            zorro_df.to_csv(outpath, index=False)
            print('made {}'.format(outpath))

        return pd.read_csv(outpath)

    elif instrument == 'NIRC2':

        namestr = 'WASP-4_20120727_NIRC2'
        outpath = (
            '../data/WASP4_NIRC2/{}_companionbounds.csv'.
            format(namestr)
        )

        if not os.path.exists(outpath):

            df = get_wasp4_mag_to_companion_contrasts()

            #
            # WASP-4_20190928_832.dat is the most contstraining curve for basically any
            # substellar mass companion. The blackbody curve works against us in 562,
            # and the seeing was better on 20190928 than 20190912.
            #
            datapath = '../data/WASP4_NIRC2/WASP-4_Kp_contrast_2012_07_27_contrast_dk_full_img.txt'
            nirc2_df = pd.read_csv(
                datapath, skiprows=2,
                names=['ang_sep', 'delta_mag', 'completeness'], delim_whitespace=True
            )

            #
            # Interpolation function to convert observed deltamags to deltamass.
            #
            fn_dmag_to_mass = interp1d(
                nparr(df.dmag_832),
                nparr(df.mass),
                kind='quadratic',
                bounds_error=False,
                fill_value=np.nan
            )

            nirc2_df['m_comp/m_sun'] = fn_dmag_to_mass(nirc2_df.delta_mag)

            nirc2_df.to_csv(outpath, index=False)
            print('made {}'.format(outpath))

        return pd.read_csv(outpath)

    elif instrument == 'HRcam':

        namestr = 'TOI837_20190723_HRcam'
        outpath = (
            '../data/speckle/{}_companionbounds.csv'.
            format(namestr)
        )

        if not os.path.exists(outpath):

            df = get_toi837_mag_to_companion_contrasts()

            datapath = '../data/speckle/sep_vs_dmag.csv'
            hrcam_df = pd.read_csv(datapath, names=['sep_arcsec','dmag'])

            #
            # Interpolation function to convert observed deltamags to deltamass.
            #
            fn_dmag_to_mass = interp1d(
                nparr(df.dmag_HRcam_Ic),
                nparr(df.mass),
                kind='quadratic',
                bounds_error=False,
                fill_value=np.nan
            )

            hrcam_df['m_comp/m_sun'] = fn_dmag_to_mass(hrcam_df.dmag)

            hrcam_df.to_csv(outpath, index=False)
            print('made {}'.format(outpath))

            dmag_smooth = np.linspace(0, 100, int(1e4))
            smooth_df = pd.DataFrame({
                'dmag_smooth': dmag_smooth,
                'm_comp/m_sun': fn_dmag_to_mass(dmag_smooth)
            })
            smooth_df.to_csv('../data/speckle/smooth_dmag_to_mass.csv',
                             index=False)

        return pd.read_csv(outpath)

    else:
        raise NotImplementedError


def main():

    ##########
    # config #
    ##########
    # "HRcam", or [from WASP4 analysis] 'both (Zorro+NIRC2)', 'Zorro', or 'NIRC2'.
    instrument = 'HRcam'

    ##############
    # end config #
    ##############

    df = get_companion_bounds(instrument)

if __name__ == "__main__":
    main()
