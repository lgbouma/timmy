from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.table import Table
import pandas as pd
from astropy.time import Time

t = Table.read('../data/gaia_csvs/gaia_vizier_asu.fits')
t['source_id'] = t['DR2Name'].astype(str)

df = t.to_pandas()

a = '5251470948229949568'

adf = df[df.source_id.str.contains(a)]

c = SkyCoord(ra=float(adf.RA_ICRS)*u.degree,
             dec=float(adf.DE_ICRS)*u.degree,
             distance=(1/(float(adf.Plx)*1e-3))*u.pc,
             pm_dec=float(adf.pmRA)*u.mas/u.yr,
             pm_ra_cosdec=float(adf.pmDE)*u.mas/u.yr,
             obstime=Time('2015-06-01 05:00:00'), # roughly gaia time
             frame='icrs')

new_obstime = Time('1982-05-16 09:43:00')
c_new = c.apply_space_motion(new_obstime=new_obstime)

print(c)
print(c_new)

print(c.separation(c_new).to(u.arcsec))
