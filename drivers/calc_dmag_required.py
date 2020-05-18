"""
Follow Vanderburg+2019 (i.e., Seager & Mallen-Ornelas 03)

Naive:
\Delta m < - \frac{5}{2} \log_{10} \delta_{\rm obs}.
Smarter:
\Delta m < -\frac{5}{2} \log_{10} (\delta_{\rm obs} t_{12}^2/t_{13}^2).

For a grazing transit, the two approaches give the same constraint.
"""

from math import log10
from astropy import units as u

delta_obs = 4e-3

t_14 = 1.83
t_13 = t_14/2  # ill-defined for grazing transits.
t_34 = t_14 - t_13 # egress time
t_12 = t_34 # ingress time, assumed symmetric.

dmag_naive = -2.5 * log10(delta_obs)

dmag_smarter = -2.5 * log10(delta_obs*t_12**2/(t_13**2))

Tmag_837 = 9.93

Tmag_naive = Tmag_837 + dmag_naive
Tmag_smarter = Tmag_837 + dmag_smarter

print('Naive: {}'.format(Tmag_naive))
print('t_14: {}hr. t_13: {}hr. t_12: {}hr.'.format(t_14, t_13, t_12))
print('Smarter: {}'.format(Tmag_smarter))
