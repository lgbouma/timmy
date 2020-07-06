from astropy import units as u, constants as c

depth = 4e-3
vsini = 17.48*u.km/u.s
b = 0.95

u1 = .37
u2 = .25
mu = (1-b**2)**(1/2)

I_factor = 1 - u1*(1-mu) - u2*(1-mu)**2

delta_v_RM = (depth * vsini * (1-b**2)**(1/2)).to(u.m/u.s)

print(f'{delta_v_RM:.2f}')
print(f'{delta_v_RM*I_factor:.2f}')
