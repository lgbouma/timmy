import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os
from astropy import units as units, constants as const

import exoplanet as xo
from exoplanet.gp import terms, GP
import theano.tensor as tt

from timmy.plotting import plot_MAP_data
from timmy.paths import RESULTSDIR

from timmy.priors import RSTAR, RSTAR_STDEV, LOGG, LOGG_STDEV

# factor * 10**logg / r_star = rho
factor = 5.141596357654149e-05

class ModelParser:

    def __init__(self, modelid):
        self.initialize_model(modelid)

    def initialize_model(self, modelid):
        self.modelid = modelid
        self.modelcomponents = modelid.split('_')
        self.verify_modelcomponents()

    def verify_modelcomponents(self):

        validcomponents = ['transit', 'gprot', 'rv']
        for i in range(5):
            validcomponents.append('{}sincosPorb'.format(i))
            validcomponents.append('{}sincosProt'.format(i))

        assert len(self.modelcomponents) >= 1

        for modelcomponent in self.modelcomponents:
            if modelcomponent not in validcomponents:
                errmsg = (
                    'Got modelcomponent {}. validcomponents include {}.'
                    .format(modelcomponent, validcomponents)
                )
                raise ValueError(errmsg)


class ModelFitter(ModelParser):
    """
    Given a modelid of the form "transit", and observed x and y values
    (typically time and flux), run the inference.

    The model implemented is of the form

    Y ~ N([Mandel-Agol transit], σ^2).
    """

    def __init__(self, modelid, x_obs, y_obs, y_err, prior_d,
                 N_samples=2000, N_cores=16, N_chains=4,
                 plotdir=None, pklpath=None, overwrite=1):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.y_err = y_err
        self.t_exp = np.nanmedian(np.diff(x_obs))

        self.initialize_model(modelid)
        self.verify_inputdata()
        #FIXME threadsafety
        self.run_inference(prior_d, pklpath, make_threadsafe=False)


    def verify_inputdata(self):
        np.testing.assert_array_equal(
            self.x_obs,
            self.x_obs[np.argsort(self.x_obs)]
        )
        assert len(self.x_obs) == len(self.y_obs)
        assert isinstance(self.x_obs, np.ndarray)
        assert isinstance(self.y_obs, np.ndarray)


    def run_inference(self, prior_d, pklpath, make_threadsafe=True):

        # if the model has already been run, pull the result from the
        # pickle. otherwise, run it.
        if os.path.exists(pklpath):
            d = pickle.load(open(pklpath, 'rb'))
            self.model = d['model']
            self.trace = d['trace']
            self.map_estimate = d['map_estimate']
            return 1

        with pm.Model() as model:

            # Fixed data errors.
            sigma = self.y_err

            # Define priors and PyMC3 random variables to sample over.

            # Stellar parameters. (Following tess.world notebooks).
            logg_star = pm.Normal("logg_star", mu=LOGG, sd=LOGG_STDEV)
            r_star = pm.Bound(pm.Normal, lower=0.0)(
                "r_star", mu=RSTAR, sd=RSTAR_STDEV
            )
            rho_star = pm.Deterministic(
                "rho_star", factor*10**logg_star / r_star
            )

            # Transit parameters.
            mean = pm.Normal(
                "mean", mu=prior_d['mean'], sd=1e-2, testval=prior_d['mean']
            )

            t0 = pm.Normal(
                "t0", mu=prior_d['t0'], sd=5e-3, testval=prior_d['t0']
            )

            period = pm.Normal(
                'period', mu=prior_d['period'], sd=5e-3,
                testval=prior_d['period']
            )

            # u = xo.distributions.QuadLimbDark(
            #     "u", testval=prior_d['u']
            # )

            # NOTE: might want to implement this, for better values
            u0 = pm.Uniform(
                'u[0]', lower=prior_d['u[0]']-0.15,
                upper=prior_d['u[0]']+0.15,
                testval=prior_d['u[0]']
            )
            u1 = pm.Uniform(
                'u[1]', lower=prior_d['u[1]']-0.15,
                upper=prior_d['u[1]']+0.15,
                testval=prior_d['u[1]']
            )
            u = [u0, u1]

            # # The Espinoza (2018) parameterization for the joint radius ratio and
            # # impact parameter distribution
            # r, b = xo.distributions.get_joint_radius_impact(
            #     min_radius=0.001, max_radius=1.0,
            #     testval_r=prior_d['r'],
            #     testval_b=prior_d['b']
            # )
            # # NOTE: apparently, it's been deprecated. I wonder why...

            log_r = pm.Uniform('log_r', lower=np.log(1e-2), upper=np.log(1),
                               testval=prior_d['log_r'])
            r = pm.Deterministic('r', tt.exp(log_r))

            b = xo.distributions.ImpactParameter(
                "b", ror=r, testval=prior_d['b']
            )

            orbit = xo.orbits.KeplerianOrbit(
                period=period, t0=t0, b=b, rho_star=rho_star
            )

            mu_transit = pm.Deterministic(
                'mu_transit',
                xo.LimbDarkLightCurve(u).get_light_curve(
                    orbit=orbit, r=r, t=self.x_obs, texp=self.t_exp
                ).T.flatten()
            )

            #
            # Derived parameters
            #

            # planet radius in jupiter radii
            r_planet = pm.Deterministic(
                "r_planet", (r*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
            )

            #
            # eq 30 of winn+2010, ignoring planet density.
            #
            a_Rs = pm.Deterministic(
                "a_Rs",
                (rho_star * period**2)**(1/3)
                *
                (( (1*units.gram/(1*units.cm)**3) * (1*units.day**2)
                  * const.G / (3*np.pi)
                )**(1/3)).cgs.value
            )

            #
            # cosi. assumes e=0 (e.g., Winn+2010 eq 7)
            #
            cosi = pm.Deterministic("cosi", b / a_Rs)

            # probably safer than tt.arccos(cosi)
            sini = pm.Deterministic("sini", pm.math.sqrt( 1 - cosi**2 ))

            #
            # transit durations (T_14, T_13) for circular orbits. Winn+2010 Eq 14, 15.
            # units: hours.
            #
            T_14 = pm.Deterministic(
                'T_14',
                (period/np.pi)*
                tt.arcsin(
                    (1/a_Rs) * pm.math.sqrt( (1+r)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            T_13 =  pm.Deterministic(
                'T_13',
                (period/np.pi)*
                tt.arcsin(
                    (1/a_Rs) * pm.math.sqrt( (1-r)**2 - b**2 )
                    * (1/sini)
                )*24
            )

            #
            # mean model and likelihood
            #

            mean_model = mu_transit + mean

            mu_model = pm.Deterministic('mu_model', mean_model)

            likelihood = pm.Normal('obs', mu=mean_model, sigma=sigma,
                                   observed=self.y_obs)

            # Optimizing
            map_estimate = pm.find_MAP(model=model)

            # start = model.test_point
            # if 'transit' in self.modelcomponents:
            #     map_estimate = xo.optimize(start=start,
            #                                vars=[r, b, period, t0])
            # map_estimate = xo.optimize(start=map_estimate)

            # Plot the simulated data and the maximum a posteriori model to
            # make sure that our initialization looks ok.
            self.y_MAP = (
                map_estimate['mean'] + map_estimate['mu_transit']
            )

            if make_threadsafe:
                pass
            else:
                # as described in
                # https://github.com/matplotlib/matplotlib/issues/15410
                # matplotlib is not threadsafe. so do not make plots before
                # sampling, because some child processes tries to close a
                # cached file, and crashes the sampler.

                print(map_estimate)

                if self.PLOTDIR is None:
                    raise NotImplementedError
                outpath = os.path.join(self.PLOTDIR,
                                       'test_{}_MAP.png'.format(self.modelid))
                plot_MAP_data(self.x_obs, self.y_obs, self.y_MAP, outpath)

            # sample from the posterior defined by this model.
            trace = pm.sample(
                tune=self.N_samples, draws=self.N_samples,
                start=map_estimate, cores=self.N_cores,
                chains=self.N_chains,
                step=xo.get_dense_nuts_step(target_accept=0.8),
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate
