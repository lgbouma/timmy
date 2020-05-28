import os
import numpy as np, pandas as pd

import radvel
from radvel.plot import orbit_plots, mcmc_plots
from radvel.mcmc import statevars
from radvel.driver import save_status, load_status
from radvel.utils import semi_amplitude

from copy import deepcopy

import configparser

from astropy import units as u, constants as const

class args_object(object):
    """
    a minimal version of the "parser" object that lets you work with the
    high-level radvel API from python. (without directly using the command line
    interface)
    """
    def __init__(self, setupfn, outputdir):
        # return args object with the following parameters set
        self.setupfn = setupfn
        self.outputdir = outputdir
        self.decorr = False
        self.plotkw = {}
        self.gp = False


def draw_models(setupfn, outputdir, chaindf, times, n_samples=None):
    # return n_samples x n_rvs RV models, and the parameters they were drawn
    # from.

    chain_samples = chaindf.sample(n=n_samples).drop('Unnamed: 0', axis=1)
    chain_sample_params = chain_samples.drop('lnprobability', axis=1)

    # get residuals, RVs, error bars, etc from the fit that has been run..
    args = args_object(setupfn, outputdir)
    args.inputdir = outputdir
    args.type = ['rv']
    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(args.inputdir, "{}_radvel.stat".format(conf_base))

    status = load_status(statfile)

    # initialize posterior object from the statfile that is passed.
    post = radvel.posterior.load(status.get('fit', 'postfile'))

    rvmodel = deepcopy(post.likelihood.model)
    rvmodel.params['curv'] = radvel.Parameter(0, vary=False)

    # set the RV model parameters use the sample parameters from the chain.
    # then calculate the model and append it.

    model_rvs = []
    for ix, sample_row in chain_sample_params.iterrows():

        for key in sample_row.index:

            rvmodel.params[key] = radvel.Parameter(sample_row[key], vary=True)

        model_rvs.append( rvmodel(times) )

    model_rvs = np.vstack(model_rvs)

    return model_rvs, chain_sample_params



def _get_fit_results(setupfn, outputdir):

    args = args_object(setupfn, outputdir)
    args.inputdir = outputdir

    # radvel plot -t rv -s $basepath
    args.type = ['rv']

    # get residuals, RVs, error bars, etc from the fit that has been run..
    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(args.inputdir, "{}_radvel.stat".format(conf_base))

    status = load_status(statfile)

    if not status.getboolean('fit', 'run'):
        raise AssertionError("Must perform max-liklihood fit before plotting")

    # initialize posterior object from the statfile that is passed.
    post = radvel.posterior.load(status.get('fit', 'postfile'))

    # update the posterior to match the median best-fit parameters.
    summarycsv = os.path.join(outputdir, "WASP4_post_summary.csv")
    sdf = pd.read_csv(summarycsv)
    for param in [c for c in sdf.columns if 'Unnamed' not in c]:
        post.params[param] = radvel.Parameter(value=sdf.ix[1][param])

    P, _ = radvel.utils.initialize_posterior(config_file)
    if hasattr(P, 'bjd0'):
        args.plotkw['epoch'] = P.bjd0

    model = post.likelihood.model
    rvtimes = post.likelihood.x
    rvs = post.likelihood.y
    rverrs = post.likelihood.errorbars()
    num_planets = model.num_planets
    telvec = post.likelihood.telvec

    dvdt_merr = sdf['dvdt'].iloc[0]
    dvdt_perr = sdf['dvdt'].iloc[2]

    rawresid = post.likelihood.residuals()

    resid = (
        rawresid + post.params['dvdt'].value*(rvtimes-model.time_base)
        + post.params['curv'].value*(rvtimes-model.time_base)**2
    )

    rvtimes, rvs, rverrs, resid, telvec = rvtimes, rvs, rverrs, resid, telvec
    dvdt, curv = post.params['dvdt'].value, post.params['curv'].value
    dvdt_merr, dvdt_perr = dvdt_merr, dvdt_perr
    time_base = model.time_base

    return (rvtimes, rvs, rverrs, resid, telvec, dvdt, curv, dvdt_merr,
            dvdt_perr, time_base)



def initialize_sim_posterior(data, mass_c, sma_c, incl_c, ecc_c,
                             gammajit_dict, verbose=True):
    """
    Initialize Posterior object to be used for the "second planet" models.

    Basically a hack of radvel.utils.initialize_posterior

    Returns:
        tuple: (object representation of config file, radvel.Posterior object)
    """

    system_name = 'simWASP4'

    mstar = 0.864*u.Msun
    mass_b = 1.186*u.Mjup
    Mtotal = (mass_c + mass_b + mstar).to(u.Msun).value

    Msini = (mass_c * np.sin(np.deg2rad(incl_c))).to(u.Mjup).value
    period_c = np.sqrt(
        4*np.pi**2 * sma_c**3 / (const.G * Mtotal*u.Msun)
    ).to(u.day).value

    k_c = semi_amplitude(Msini, period_c, Mtotal, ecc_c, Msini_units='jupiter')

    P = templateWASP4(data, period_c, ecc_c, k_c, tc_c=2455470,
                      w_c=0.42*np.pi/2, gammajit_dict=gammajit_dict)

    # initalization from radvel.utils.initialize_posterior

    params = P.params
    assert str(params.basis) == "Basis Object <{}>".format(P.fitting_basis), """
            Parameters in config file must be converted to fitting basis.
            """

    decorr = False
    decorr_vars = []

    for key in params.keys():
        if key.startswith('logjit'):
            msg = """
            Fitting log(jitter) is depreciated. Please convert your config
            files to initialize 'jit' instead of 'logjit' parameters.
            Converting 'logjit' to 'jit' for you now.
            """
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            newkey = key.replace('logjit', 'jit')
            params[newkey] = (
                radvel.model.Parameter(value=np.exp(params[key].value),
                                       vary=params[key].vary)
            )
            del params[key]

    iparams = radvel.basis._copy_params(params)

    # Make sure we don't have duplicate indicies in the DataFrame
    P.data = P.data.reset_index(drop=True)

    # initialize RVmodel object
    mod = radvel.RVModel(params, time_base=P.time_base)

    # initialize Likelihood objects for each instrument
    telgrps = P.data.groupby('tel').groups
    likes = {}
    for inst in P.instnames:
        assert inst in P.data.groupby('tel').groups.keys(), \
            "No data found for instrument '{}'.\nInstruments found in this dataset: {}".format(inst, list(telgrps.keys()))
        decorr_vectors = {}
        if decorr:
            for d in decorr_vars:
                decorr_vectors[d] = P.data.iloc[telgrps[inst]][d].values

        try:
            hnames = P.hnames[inst]
            liketype = radvel.likelihood.GPLikelihood
            try:
                kernel_name = P.kernel_name[inst]
                # if kernel_name == "Celerite":
                #     liketype = radvel.likelihood.CeleriteLikelihood
                if kernel_name == "Celerite":
                     liketype = radvel.likelihood.CeleriteLikelihood
            except AttributeError:
                kernel_name = "QuasiPer"
        except AttributeError:
            liketype = radvel.likelihood.RVLikelihood
            kernel_name = None
            hnames = None
        likes[inst] = liketype(
            mod, P.data.iloc[telgrps[inst]].time,
            P.data.iloc[telgrps[inst]].mnvel,
            P.data.iloc[telgrps[inst]].errvel, hnames=hnames, suffix='_'+inst,
            kernel_name=kernel_name, decorr_vars=decorr_vars,
            decorr_vectors=decorr_vectors
        )
        likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
        likes[inst].params['jit_'+inst] = iparams['jit_'+inst]

    like = radvel.likelihood.CompositeLikelihood(list(likes.values()))

    # Initialize Posterior object
    post = radvel.posterior.Posterior(like)
    post.priors = P.priors

    #return P, post
    return post


class templateWASP4(object):
    """
    a version of the "initialization module" with argument enabled to allow
    simulated "second planet" fits
    """

    def __init__(self, data, period_c, ecc_c, k_c, tc_c=2455470,
                 w_c=0.42*np.pi/2, gammajit_dict=None):

        # for commenting and definitions, see src/WASP4.py

        starname = 'simWASP4'
        nplanets = 1    # number of planets in the system
        instnames = ['CORALIE', 'HARPS', 'HIRES']   # list of instrument names.
        ntels = len(instnames)
        bjd0 = 0   # this number has been subtracted off your timestamps
        planet_letters = {1: 'c'}  # map numbers in Parameters keys to letters

        anybasis_params = radvel.Parameters(nplanets, basis='per tc e w k',
                                            planet_letters=planet_letters)

        anybasis_params['per1'] = radvel.Parameter(value=period_c)
        anybasis_params['tc1'] = radvel.Parameter(value=tc_c)
        anybasis_params['e1'] = radvel.Parameter(value=ecc_c)
        anybasis_params['w1'] = radvel.Parameter(value=w_c)
        anybasis_params['k1'] = radvel.Parameter(value=k_c)

        time_base = 2455470
        anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
        anybasis_params['curv'] = radvel.Parameter(value=0.0)

        # set gamma_CORALIE, gamma_HARPS, jit_HARPS, etc
        for k in gammajit_dict:
            anybasis_params[k] = radvel.Parameter(
                value=gammajit_dict[k], vary=False
            )

        # Convert input orbital parameters into the fitting basis
        fitting_basis = 'per tc e w k' # see radvel.basis.BASIS_NAMES
        params = (
            anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)
        )

        # Set the 'vary' attributes of each of the parameters in the fitting
        # basis. A parameter's 'vary' attribute should be set to False if you
        # wish to hold it fixed during the fitting process. By default, all
        # 'vary' parameters are set to True.

        params['tc1'].vary = True
        params['w1'].vary = True
        # params['secosw1'].vary = True
        # params['sesinw1'].vary = True

        params['per1'].vary = False  # if false, struggles more w/ convergence.
        params['k1'].vary = False
        params['dvdt'].vary = False
        params['curv'].vary = False
        params['e1'].vary = False

        # Define prior shapes and widths here.
        priors = [
            radvel.prior.HardBounds(
                'w1', 0.0, np.pi
            ),
            radvel.prior.HardBounds(
                'tc1',
                params['tc1'].value-period_c/2,
                params['tc1'].value+period_c/2
            )
        ]

        # optional argument that can contain stellar mass in solar units
        # (mstar) and uncertainty (mstar_err). If not set, mstar will be set to
        # nan.
        stellar = dict(mstar=0.864, mstar_err=0.0087)
        planet = dict(rp1=1.321, rperr1=0.039)

        self.params = params
        self.priors = priors
        self.stellar = stellar
        self.planet = planet
        self.time_base = time_base
        self.starname = starname
        self.nplanets = nplanets
        self.instnames = instnames
        self.ntels = ntels
        self.fitting_basis = fitting_basis
        self.bjd0 = bjd0
        self.planet_letters = planet_letters
        self.data = data
