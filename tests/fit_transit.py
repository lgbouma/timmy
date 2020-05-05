"""
Y ~ N([Mandel-Agol transit], σ^2).
Errors are treated as fixed and known observables.
In this example, two transits are fitted simultaneously for "period", "t0",
"r", "b", "u" (quadratic). It's a 12 parameter fit. It takes about 1 minute.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe
import exoplanet as xo
from billy.models import sin_model
from billy.plotting import plot_test_data, savefig

# randomly sample some periods and phases and then define the time sampling
modelid = 'transit'
np.random.seed(123)
periods = np.random.uniform(5, 20, 2)
t0s = periods * np.random.rand(2)
rs = np.array([0.04, 0.06])
bs = np.random.rand(2)
us = np.array([0.3, 0.2])

t = np.arange(0, 80, 0.02)
yerr = 5e-4
texp = 30/(60*24)

pklpath = os.path.join(os.path.expanduser('~'), 'local', 'billy',
                       'model_{}.pkl'.format(modelid))

sampled = 0
if not os.path.exists(pklpath):

    with pm.Model() as model:
        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0)
        # The time of a reference transit for each planet
        t0 = pm.Normal("t0", mu=t0s, sd=1.0, shape=2)
        # The log period; also tracking the period itself
        logP = pm.Normal("logP", mu=np.log(periods), sd=0.1, shape=2)
        period = pm.Deterministic("period", pm.math.exp(logP))
        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        u = xo.distributions.QuadLimbDark("u", testval=us)
        r = pm.Uniform(
            "r", lower=0.01, upper=0.1, shape=2, testval=rs
        )
        b = xo.distributions.ImpactParameter(
            "b", ror=r, shape=2, testval=bs
        )

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=t, texp=texp
        )
        light_curve = pm.math.sum(light_curves, axis=-1) + mean
        # Here we track the value of the model light curve for plotting
        # purposes
        pm.Deterministic("light_curves", light_curves)
        # In this line, we simulate the dataset that we will fit
        y = xo.eval_in_model(light_curve)
        y += yerr * np.random.randn(len(y))
        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)
        # Fit for the maximum a posteriori parameters given the simuated
        # dataset
        map_estimate = xo.optimize(start=model.test_point)

    # plot the simulated data and the maximum a posteriori model to make sure that
    # our initialization looks ok.
    plt.plot(t, y, ".k", ms=4, label="data")
    for i, l in enumerate("bc"):
        plt.plot(
            t, map_estimate["light_curves"][:, i], lw=1, label="planet {0}".format(l)
        )
    plt.xlim(t.min(), t.max())
    plt.ylabel("relative flux")
    plt.xlabel("time [days]")
    plt.legend(fontsize=10)
    _ = plt.title("map model")
    fig = plt.gcf()
    savefig(fig, '../results/test_results/test_{}_MAP.png'.format(modelid), writepdf=0)

    # sample from the posterior defined by this model. As usual, there are strong
    # covariances between some of the parameters so we’ll use
    # exoplanet.get_dense_nuts_step().
    np.random.seed(42)
    with model:
        trace = pm.sample(
            tune=3000,
            draws=3000,
            start=map_estimate,
            cores=16,
            chains=4,
            step=xo.get_dense_nuts_step(target_accept=0.9),
        )

    with open(pklpath, 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace,
                     'map_estimate': map_estimate}, buff)

    samples = pm.trace_to_dataframe(trace, varnames=["period", "r"])
    truth = np.concatenate(
        xo.eval_in_model([period, r], model.test_point, model=model)
    )
    fig = corner.corner(
        samples,
        truths=truth,
        labels=["period 1", "period 2", "radius 1", "radius 2"],
    )
    fig.savefig('../results/test_results/test_{}_corner_subset.png'.format(modelid))

    sampled = 1

else:
    d = pickle.load(open(pklpath, 'rb'))
    model, trace, map_estimate = d['model'], d['trace'], d['map_estimate']

print(pm.summary(trace, varnames=["period", "t0", "r", "b", "u", "mean"]))

true_d = OrderedDict(
    {'mean':1, 't0__0':t0s[0], 't0__1':t0s[1], 'period__0':periods[0],
     'period__1':periods[1], 'u__0':us[0], 'u__1':us[1], 'r__0':rs[0],
     'r__1':rs[1], 'b__0':bs[0], 'b__1':bs[1]}
)

trace_df = trace_to_dataframe(trace, varnames=["period", "t0", "r", "b", "u", "mean"])
truths = [true_d[k] for k in true_d.keys()]
fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    title_kwargs={"fontsize": 12}, truths=truths)
fig.savefig('../results/test_results/test_{}_corner.png'.format(modelid))

# phase plots
if sampled:
    for n, letter in enumerate("bc"):
        plt.figure()

        # Get the posterior median orbital parameters
        p = np.median(trace["period"][:, n])
        t0 = np.median(trace["t0"][:, n])

        # Compute the median of posterior estimate of the contribution from
        # the other planet. Then we can remove this from the data to plot
        # just the planet we care about.
        other = np.median(trace["light_curves"][:, :, (n + 1) % 2], axis=0)

        # Plot the folded data
        x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
        plt.errorbar(
            x_fold, y - other, yerr=yerr, fmt=".k", label="data", zorder=-1000
        )

        # Plot the folded model
        inds = np.argsort(x_fold)
        inds = inds[np.abs(x_fold)[inds] < 0.3]
        pred = trace["light_curves"][:, inds, n] + trace["mean"][:, None]
        pred = np.median(pred, axis=0)
        plt.plot(x_fold[inds], pred, color="C1", label="model")

        # Annotate the plot with the planet's period
        txt = "period = {0:.4f} +/- {1:.4f} d".format(
            np.mean(trace["period"][:, n]), np.std(trace["period"][:, n])
        )
        plt.annotate(
            txt,
            (0, 0),
            xycoords="axes fraction",
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=12,
        )

        plt.legend(fontsize=10, loc=4)
        plt.xlim(-0.5 * p, 0.5 * p)
        plt.xlabel("time since transit [days]")
        plt.ylabel("relative flux")
        plt.title("planet {0}".format(letter))
        plt.xlim(-0.3, 0.3)
        plt.savefig('../results/test_results/test_{}_phase_{}.png'.format(modelid, letter))
        plt.close('all')
