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

t = np.sort(
    np.append(np.random.uniform(0, 3.8, 57), np.random.uniform(5.5, 10, 68))
)  # The input coordinates must be sorted
yerr = np.random.uniform(0.08, 0.22, len(t))
y = (
    0.2 * (t - 5)
    + np.sin(3 * t + 0.1 * (t - 5) ** 2)
    + yerr * np.random.randn(len(t))
)

true_t = np.linspace(0, 10, 5000)
true_y = 0.2 * (true_t - 5) + np.sin(3 * true_t + 0.1 * (true_t - 5) ** 2)

plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="data")
plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="truth")
plt.legend(fontsize=12)
plt.xlabel("t")
plt.ylabel("y")
plt.xlim(0, 10)
_ = plt.ylim(-2.5, 2.5)
plt.savefig('../results/test_results/gp_model/data.png', dpi=200)
plt.close('all')

pklpath = os.path.join(os.path.expanduser('~'), 'local', 'timmy',
                       'model_gp.pkl')

if os.path.exists(pklpath):
    d = pickle.load(open(pklpath, 'rb'))
    model, trace, map_estimate = d['model'], d['trace'], d['map_estimate']

else:

    with pm.Model() as model:

        mean = pm.Normal("mean", mu=0.0, sigma=1.0)
        S1 = pm.InverseGamma(
            "S1", **estimate_inverse_gamma_parameters(0.5 ** 2, 10.0 ** 2)
        )
        S2 = pm.InverseGamma(
            "S2", **estimate_inverse_gamma_parameters(0.25 ** 2, 1.0 ** 2)
        )
        w1 = pm.InverseGamma(
            "w1", **estimate_inverse_gamma_parameters(2 * np.pi / 10.0, np.pi)
        )
        w2 = pm.InverseGamma(
            "w2", **estimate_inverse_gamma_parameters(0.5 * np.pi, 2 * np.pi)
        )
        log_Q = pm.Uniform("log_Q", lower=np.log(2), upper=np.log(10))

        # Set up the kernel an GP
        kernel = terms.SHOTerm(S_tot=S1, w0=w1, Q=1.0 / np.sqrt(2))
        kernel += terms.SHOTerm(S_tot=S2, w0=w2, log_Q=log_Q)
        gp = GP(kernel, t, yerr ** 2, mean=mean)

        # Condition the GP on the observations and add the marginal likelihood
        # to the model
        gp.marginal("gp", observed=y)


    with model:
        map_soln = xo.optimize(start=model.test_point)


    with model:
        mu, var = xo.eval_in_model(
            gp.predict(true_t, return_var=True, predict_mean=True), map_soln
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
    plt.xlim(0, 10)
    _ = plt.ylim(-2.5, 2.5)
    plt.savefig('../results/test_results/gp_model/prediction_uncert.png', dpi=200)
    plt.close('all')


    with model:
        trace = pm.sample(
            tune=2000,
            draws=2000,
            start=map_soln,
            cores=2,
            chains=2,
            step=xo.get_dense_nuts_step(target_accept=0.9),
        )

    print(pm.summary(trace))

    # truth = np.concatenate(
    #     xo.eval_in_model([period, r], model.test_point, model=model)
    # )
    fig = corner.corner(
        samples,
        # truths=truth,
        labels=["S1", "S2", "w1", "w2", "log_Q"],
    )
    fig.savefig('../results/test_results/gp_model/test_gp_corner.png')
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
    plt.xlim(0, 10)
    _ = plt.ylim(-2.5, 2.5)
    plt.savefig('../results/test_results/gp_model/test_gp_sampling.png')
    plt.close('all')
