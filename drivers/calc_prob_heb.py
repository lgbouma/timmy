import numpy as np, matplotlib.pyplot as plt
from scipy.stats import lognorm
from astropy import units as u, constants as const

x = np.logspace(-2,10,100)

mu_logP = 5.03
sigma_lopP = 2.28

sample_logP = np.random.normal(loc=mu_logP, scale=sigma_lopP, size=int(1e4))

P = 10**(sample_logP)*u.day
sma = ((
    P**2 * const.G*(1*u.Msun) / (4*np.pi**2)
)**(1/3)).to(u.au).value

sample_loga = np.log10(sma)

sel = (sample_loga > 1) & (sample_loga < 2)

N_tot = len(sample_loga)
frac_q = 0.1 # roughly 0.1-0.2 Msun is allowed
frac_a = len(sample_loga[sel])/len(sample_loga)

frac_bin = frac_a*frac_q
print(frac_bin)


# plt.close('all')
# f,ax = plt.subplots(figsize=(4,3))
# h = ax.hist(sample_logP, bins=50)
# ax.set_xlabel('logP')
# f.savefig('sample_logP.png', bbox_inches='tight')
# 
# plt.close('all')
# f,ax = plt.subplots(figsize=(4,3))
# h = ax.hist(sample_loga, bins=50)
# ax.set_xlabel('loga')
# f.savefig('sample_loga.png', bbox_inches='tight')
