"""
Y ~ N([Mandel-Agol transit] + A * np.sin(ω*t + φ_0), σ^2).
Errors are treated as fixed and known observables.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner
from collections import OrderedDict

from pymc3.backends.tracetab import trace_to_dataframe
import exoplanet as xo

from billy.models import sin_model, transit_model
from billy.plotting import plot_test_data, savefig
from billy.convenience import flatten as bflatten

#################
# generate data #
#################
np.random.seed(42)
modelid = 'transit_sinusoid'
N_samples = 3000
N_cores, N_chains = 16, 4
true_sigma = 1e-4

t_exp = 30/(60*24)
mstar, rstar = 1, 1
tra_d = OrderedDict({'period':4.3, 't0':0.2, 'r':0.04, 'b':0.5,
                     'u':[0.3,0.2], 'mean':0})
sin_d = OrderedDict({'A0': 0.01, 'omega0': 0.3, 'phi0': 0.3141})
true_d = {**tra_d, **sin_d}

transit_params = [tra_d[k] for k in tra_d.keys()]
sin_params = [sin_d[k] for k in sin_d.keys()]
true_params = [true_d[k] for k in true_d.keys()]

np.random.seed(42)
x_obs = np.arange(0, 28, t_exp)
y_mod = (
    sin_model(sin_params, x_obs) +
    transit_model(transit_params, x_obs, mstar=mstar, rstar=rstar)
)
y_obs = y_mod + np.random.normal(scale=true_sigma, size=len(x_obs))

plot_test_data(x_obs, y_obs, y_mod, modelid, outdir='../results/test_results/')

#############################
# fit and sample parameters #
#############################

pklpath = os.path.join(os.path.expanduser('~'), 'local', 'billy',
                       'model_{}.pkl'.format(modelid))

if not os.path.exists(pklpath):
    with pm.Model() as model:
        sigma = true_sigma

        # Define priors
        A0 = pm.Uniform('A0', lower=0.005, upper=0.015, testval=true_d['A0'])
        omega0 = pm.Uniform('omega0', lower=0.25, upper=0.35,
                            testval=true_d['omega0'])
        phi0 = pm.Uniform('phi0', lower=0, upper=np.pi, testval=true_d['phi0'])

        mean = pm.Normal("mean", mu=0.0, sd=1.0, testval=true_d['mean'])
        t0 = pm.Uniform("t0", lower=0, upper=true_d['period'],
                        testval=true_d['t0'])
        logP = pm.Normal("logP", mu=np.log(true_d['period']), sd=0.1,
                         testval=np.log(true_d['period']))
        period = pm.Deterministic("period", pm.math.exp(logP))
        u = xo.distributions.QuadLimbDark("u", testval=true_d['u'])
        r = pm.Uniform("r", lower=0.02, upper=0.06,  testval=true_d['r'])
        b = xo.distributions.ImpactParameter("b", ror=r, testval=true_d['b'])

        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b,
                                         mstar=mstar, rstar=rstar)
        light_curve = (
            mean +
            xo.LimbDarkLightCurve(u).get_light_curve(
                orbit=orbit, r=r, t=x_obs, texp=t_exp
            )
        )
        pm.Deterministic("light_curve", light_curve)

        # Define likelihood
        mu_model = (
            light_curve.flatten()
            +
            sin_model([A0, omega0, phi0], x_obs)
        )

        likelihood = pm.Normal('obs', mu=mu_model, sigma=sigma, observed=y_obs)

        # Get MAP estimate from model.
        map_estimate = pm.find_MAP(model=model)

        # plot the simulated data and the maximum a posteriori model to make sure that
        # our initialization looks ok.
        y_MAP = (
            sin_model(
                [map_estimate[k] for k in ['A0','omega0','phi0']], x_obs
            )
            +
            map_estimate["light_curve"].flatten()
        )
        plt.plot(x_obs, y_obs, ".k", ms=4, label="data")
        plt.plot(x_obs, y_MAP, lw=1)
        plt.ylabel("relative flux")
        plt.xlabel("time [days]")
        _ = plt.title("map model")
        fig = plt.gcf()
        savefig(fig, '../results/test_results/test_{}_MAP.png'.format(modelid), writepdf=0)

        # sample from the posterior defined by this model.
        trace = pm.sample(
            tune=N_samples, draws=N_samples, start=map_estimate, cores=N_cores,
            chains=N_chains, step=xo.get_dense_nuts_step(target_accept=0.9),
        )

    with open(pklpath, 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace,
                     'map_estimate': map_estimate}, buff)

else:
    d = pickle.load(open(pklpath, 'rb'))
    model, trace, map_estimate = d['model'], d['trace'], d['map_estimate']

##################
# analyze output #
##################

print(pm.summary(trace, varnames=list(true_d.keys())))

trace_df = trace_to_dataframe(trace, varnames=list(true_d.keys()))

truths = [true_d[k] for k in true_d.keys()]
truths = list(bflatten(truths))
fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    title_kwargs={"fontsize": 12}, truths=truths)
fig.savefig('../results/test_results/test_{}_corner.png'.format(modelid))
