from exoplanet.gp import terms, GP
from exoplanet.distributions import estimate_inverse_gamma_parameters

import exoplanet as xo
import corner, os, pickle

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe
import exoplanet as xo

np.random.seed(42)

mean = 1
amp = 1.5e-2
P_rot = 2.4
w_rot = 2*np.pi/P_rot
amp_mix = 0.7
phase_off = 2.1

true_d = {}
true_d['mean'] = mean
true_d['period'] = P_rot
true_d['amp'] = amp
true_d['mix'] = amp_mix
true_d['log_Q0'] = np.nan
true_d['log_deltaQ'] = np.nan

t = np.sort(
    np.concatenate([np.random.uniform(0, 3.8, 57),
                   np.random.uniform(5.5, 10, 68),
                   np.random.uniform(11, 16.8, 57),
                   np.random.uniform(19, 25, 68)])
)  # The input coordinates must be sorted
yerr = amp*np.random.uniform(0.08, 0.22, len(t))

y = (
    mean +
    + amp*np.sin(w_rot * t )
    + amp*amp_mix*np.sin(2*w_rot * t + phase_off)
    + yerr * np.random.randn(len(t))
)

true_t = np.linspace(0, 25, 5000)
true_y = (
    mean +
    + amp*np.sin(w_rot * true_t )
    + amp*amp_mix*np.sin(2*w_rot * true_t + phase_off)
)

plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")
plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="truth")
plt.legend(fontsize=12)
plt.xlabel("t")
plt.ylabel("y")
plt.savefig('../results/test_results/rot_model/data.png', dpi=200)
plt.close('all')

pklpath = os.path.join(os.path.expanduser('~'), 'local', 'timmy',
                       'model_rot.pkl')

if os.path.exists(pklpath):
    d = pickle.load(open(pklpath, 'rb'))
    model, trace, map_estimate, gp = (
        d['model'], d['trace'], d['map_estimate'], d['gp']
    )

else:

    with pm.Model() as model:

        # NOTE: a more principled prior might be acquired using
        # "estimate_inverse_gamma_parameters"

        mean = pm.Normal("mean", mu=0.0, sigma=1.0)

        period = pm.Normal("period", mu=2.4, sigma=1.0)

        amp = pm.Uniform("amp", lower=5e-3, upper=2.5e-2)

        # This approach has (at least) three nuisance parameters. The mixing
        # amplitude between modes, "mix".  Q0 or log_Q0 (tensor) – The quality
        # factor (or really the quality factor minus one half) for the
        # secondary oscillation.  deltaQ or log_deltaQ (tensor) – The
        # difference between the quality factors of the first and the second
        # modes. This parameterization (if deltaQ > 0) ensures that the primary
        # mode alway has higher quality.  Note for the simple case of two
        # sinusoids, for log_deltaQ to work requires going between nplog(1e-1)
        # to nplog(1e20). And it does not run fast!
        mix = pm.Uniform("mix", lower=0, upper=1)

        log_Q0 = pm.Uniform("log_Q0", lower=np.log(2), upper=np.log(1e10))

        log_deltaQ = pm.Uniform("log_deltaQ",
                                lower=np.log(1e-1),
                                upper=np.log(1e10))

        kernel = terms.RotationTerm(
            amp=amp,
            period=period,
            mix=mix,
            log_Q0=log_Q0,
            log_deltaQ=log_deltaQ,
        )

        gp = GP(kernel, t, yerr**2, mean=mean)

        # Condition the GP on the observations and add the marginal likelihood
        # to the model
        gp.marginal("gp", observed=y)


    with model:
        map_estimate = xo.optimize(start=model.test_point)


    with model:
        mu, var = xo.eval_in_model(
            gp.predict(true_t, return_var=True, predict_mean=True), map_estimate
        )


    # Plot the prediction and the 1-sigma uncertainty
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")
    plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="truth")

    sd = np.sqrt(var)
    art = plt.fill_between(true_t, mu + sd, mu - sd, color="C1", alpha=0.3)
    art.set_edgecolor("none")
    plt.plot(true_t, mu, color="C1", label="prediction")

    plt.legend(fontsize=12)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.savefig('../results/test_results/rot_model/prediction_uncert.png', dpi=200)
    plt.close('all')


    with model:
        trace = pm.sample(
            tune=2000,
            draws=2000,
            start=map_estimate,
            cores=2,
            chains=2,
            step=xo.get_dense_nuts_step(target_accept=0.9),
        )

    with open(pklpath, 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace,
                     'map_estimate': map_estimate, 'gp':gp}, buff)

print(pm.summary(trace))

samples = pm.trace_to_dataframe(trace)
truths = [true_d[k] for k in list(samples.columns)]
fig = corner.corner(
    samples,
    labels=list(samples.columns),
    truths=truths
)
fig.savefig('../results/test_results/rot_model/test_rot_corner.png')
plt.close('all')


#
# Generate 50 realizations of the prediction sampling randomly from the chain
#
N_pred = 50
pred_mu = np.empty((N_pred, len(true_t)))
pred_var = np.empty((N_pred, len(true_t)))
with model:
    pred = gp.predict(true_t, return_var=True, predict_mean=True)
    for i, sample in enumerate(xo.get_samples_from_trace(trace, size=N_pred)):
        pred_mu[i], pred_var[i] = xo.eval_in_model(pred, sample)

# Plot the predictions
for i in range(len(pred_mu)):
    mu = pred_mu[i]
    sd = np.sqrt(pred_var[i])
    label = None if i else "prediction"
    art = plt.fill_between(true_t, mu + sd, mu - sd, color="C1", alpha=0.1)
    art.set_edgecolor("none")
    plt.plot(true_t, mu, color="C1", label=label, alpha=0.1)

plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")
plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="truth")
plt.legend(fontsize=12, loc=2)
plt.xlabel("t")
plt.ylabel("y")
plt.savefig('../results/test_results/rot_model/test_rot_sampling.png')
plt.close('all')
