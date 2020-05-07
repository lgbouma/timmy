from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.table import Table
import pandas as pd

t = Table.read('../data/gaia_vizier_asu.fits')
t['source_id'] = t['DR2Name'].astype(str)

df = t.to_pandas()

a = '5251470948229949568'
b = '5251470948222139904' # nearest nbhr!

adf = df[df.source_id.str.contains(a)]
bdf = df[df.source_id.str.contains(b)]

c_a = SkyCoord(ra=float(adf.RA_ICRS)*u.degree,
               dec=float(adf.DE_ICRS)*u.degree,
               distance=(1/(float(adf.Plx)*1e-3))*u.pc, frame='icrs')
c_b = SkyCoord(ra=float(bdf.RA_ICRS)*u.degree,
               dec=float(bdf.DE_ICRS)*u.degree,
               distance=(1/(float(bdf.Plx)*1e-3))*u.pc, frame='icrs')

sep = c_a.separation_3d(c_b)

print(sep.to(u.pc))

# the uncertainty will be dominated by the 1.5% parallax uncertainty
# of the secondary companion
rel_unc = float(bdf.e_Plx) / float(bdf.Plx)
sep_unc = rel_unc * sep.to(u.pc)
print(sep_unc)
