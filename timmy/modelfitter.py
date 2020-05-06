import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os

import exoplanet as xo
from exoplanet.gp import terms, GP
import theano.tensor as tt

from timmy.plotting import plot_MAP_data
from timmy.paths import RESULTSDIR

class ModelParser:

    def __init__(self, modelid):
        self.initialize_model(modelid)

    def initialize_model(self, modelid):
        self.modelid = modelid
        self.modelcomponents = modelid.split('_')
        self.verify_modelcomponents()

    def verify_modelcomponents(self):

        validcomponents = ['transit', 'gprot']
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
    Given a modelid of the form "transit_gprot", and observed x and y values
    (typically time and flux), run the inference.

    The model implemented is of the form

    Y ~ N([Mandel-Agol transit] + 2-SHO GP, σ^2).
    """

    # FIXME: N_samples 2000 standard, N_cores 16, N_chains 4
    def __init__(self, modelid, x_obs, y_obs, y_err, prior_d, mstar=1,
                 rstar=1, N_samples=1000, N_cores=16, N_chains=4,
                 plotdir=None, pklpath=None, overwrite=1):

        self.N_samples = N_samples
        self.N_cores = N_cores
        self.N_chains = N_chains
        self.PLOTDIR = plotdir
        self.OVERWRITE = overwrite
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.y_err = y_err
        self.mstar = mstar
        self.rstar = rstar
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
            # Start with the transit parameters.
            mean = pm.Normal(
                "mean",
                mu=prior_d['mean'],
                sd=1e-2,
                testval=prior_d['mean']
            )

            t0 = pm.Normal(
                "t0", mu=prior_d['t0'], sd=5e-3, testval=prior_d['t0']
            )

            period = pm.Normal(
                'period', mu=prior_d['period'], sd=5e-3,
                testval=prior_d['period']
            )

            u = xo.distributions.QuadLimbDark(
                "u", testval=prior_d['u']
            )

            r = pm.Normal(
                "r", mu=prior_d['r'], sd=0.20*prior_d['r'],
                testval=prior_d['r']
            )

            b = xo.distributions.ImpactParameter(
                "b", ror=r, testval=prior_d['b']
            )

            orbit = xo.orbits.KeplerianOrbit(
                period=period, t0=t0, b=b, mstar=self.mstar,
                rstar=self.rstar
            )

            mu_transit = pm.Deterministic(
                'mu_transit',
                xo.LimbDarkLightCurve(u).get_light_curve(
                    orbit=orbit, r=r, t=self.x_obs, texp=self.t_exp
                ).T.flatten()
            )

            mean_model = mu_transit + mean

            if self.modelcomponents == ['transit']:
                mu_model = pm.Deterministic('mu_model', mean_model)

            if 'gprot' in self.modelcomponents:

                # Instantiate the GP parameters.

                P_rot = pm.Normal("P_rot", mu=prior_d['P_rot'], sigma=1.0,
                                  testval=prior_d['P_rot'])

                amp = pm.Uniform('amp', lower=5, upper=40, testval=10)

                mix = xo.distributions.UnitUniform("mix")

                log_Q0 = pm.Normal("log_Q0", mu=1.0, sd=10.0,
                                   testval=prior_d['log_Q0'])

                log_deltaQ = pm.Normal("log_deltaQ", mu=2.0, sd=10.0,
                                       testval=prior_d['log_deltaQ'])

                kernel = terms.RotationTerm(
                    period=P_rot,
                    amp=amp,
                    mix=mix,
                    log_Q0=log_Q0,
                    log_deltaQ=log_deltaQ,
                )

                gp = GP(kernel, self.x_obs, sigma**2, mean=mean_model)

                # Condition the GP on the observations and add the marginal likelihood
                # to the model. Needed before calling "gp.predict()".
                # NOTE: This formally is the definition of the likelihood?  It
                # would be good to figure out how this works under the hood...
                gp.marginal("transit_obs", observed=self.y_obs)

                # Compute the mean model prediction for plotting purposes
                mu_gprot = pm.Deterministic("mu_gprot", gp.predict())
                mu_model = pm.Deterministic("mu_model", mu_gprot + mean_model)

            # Optimizing
            start = model.test_point
            if 'transit' in self.modelcomponents:
                map_estimate = xo.optimize(start=start,
                                           vars=[r, b, period, t0])
            if 'gprot' in self.modelcomponents:
                map_estimate = xo.optimize(start=map_estimate,
                                           vars=[P_rot, amp, mix, log_Q0, log_deltaQ])
            map_estimate = xo.optimize(start=map_estimate)
            # map_estimate = pm.find_MAP(model=model)

            # Plot the simulated data and the maximum a posteriori model to
            # make sure that our initialization looks ok.
            self.y_MAP = (
                map_estimate['mean'] + map_estimate['mu_transit']
            )
            if 'gprot' in self.modelcomponents:
                self.y_MAP += map_estimate['mu_gprot']

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
                step=xo.get_dense_nuts_step(target_accept=0.9),
            )

        with open(pklpath, 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace,
                         'map_estimate': map_estimate}, buff)

        self.model = model
        self.trace = trace
        self.map_estimate = map_estimate
