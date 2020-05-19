"""
I queried TIC8, and asked MAST "how many T<16 stars are within a radius of 360
arcseconds of TOI 837?". The answer was 298.


"""
from math import pi

r = 360  # arcseconds
Nstar = 298  # 

Nstar_per_arcsecsq = Nstar/(pi*r**2)

r_min = 0.3  # at 0.3 arcseconds, SOAR stops getting decent constraints.
r_max = 2.0  # at 2.0 arcseconds, SOAR achieves dmag(I) of 6.
allowed_area = (pi*(r_min**2))

Nstar_expected = allowed_area*Nstar_per_arcsecsq

print(f'{Nstar_expected:.2e}')

allowed_bigarea = (pi*(r_max**2))
Nstar_expected_big = allowed_bigarea*Nstar_per_arcsecsq

print(f'{Nstar_expected_big:.2e}')
