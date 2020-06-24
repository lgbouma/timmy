from radvel import driver
import os
import emcee
if not emcee.__version__ == "2.2.1":
    raise AssertionError('radvel requires emcee v2')

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

# setupfn = "/home/luke/Dropbox/proj/timmy/drivers/radvel_drivers/TOI837.py"                         # planet fits
setupfn = "/home/luke/Dropbox/proj/timmy/drivers/radvel_drivers/TOI837_fpscenario_limits.py"        # fpscenarios

# outputdir = "/home/luke/Dropbox/proj/timmy/results/radvel_fitting/20200525_simple_planet"
# outputdir = "/home/luke/Dropbox/proj/timmy/results/radvel_fitting/20200525_fpscenario"
# outputdir = "/home/luke/Dropbox/proj/timmy/results/radvel_fitting/20200624_simple_planet"
outputdir = "/home/luke/Dropbox/proj/timmy/results/radvel_fitting/20200624_fpscenario"

if not os.path.exists(outputdir):
    os.mkdir(outputdir)

args = args_object(setupfn, outputdir)

# # perform max-likelihood fit. usually needed to be done first.
# radvel fit -s $basepath
driver.fit(args)

# # plot the maxlikelihood fit
# radvel plot -t rv -s $basepath
args.type = ['rv']
driver.plots(args)

# # perform mcmc to get uncertainties
# radvel mcmc -s $basepath
args.nsteps = 10000  # Number of steps per chain [10000]
args.nwalkers = 50   # Number of walkers. [50]
args.ensembles = 16   # Number of ensembles. Will be run in parallel on separate CPUs [8]
args.maxGR = 1.01    # Maximum G-R statistic for chains to be deemed well-mixed and halt the MCMC run [1.01]
args.burnGR = 1.03   # Maximum G-R statistic to stop burn-in period [1.03]
args.minTz = 1000    # Minimum Tz to consider well-mixed [1000]
args.minsteps = 1000 # Minimum number of steps per walker before convergence tests are performed [1000].
                     # Convergence checks will start after the minsteps threshold or the minpercent threshold has been hit.
args.minpercent = 5  # Minimum percentage of steps before convergence tests are performed [5]
                     # Convergence checks will start after the minsteps threshold or the minpercent threshold has been hit.
args.thin = 1        # Save one sample every N steps [default=1, save all samples]
args.serial = False  # If True, run MCMC in serial instead of parallel. [False]
driver.mcmc(args)

# # corner plot the samples
# radvel plot -t rv corner trend -s $basepath
args.type = ['rv','corner','trend']
driver.plots(args)

# # make a sick pdf report
# radvel report -s $basepath
args.comptype= 'ic' # Type of model comparison table to include. Default: ic
args.latex_compiler = 'pdflatex' # path to latex compiler
# driver.report(args)

# # optionally, include stellar parameters to derive physical parameters for the
# # planetary system
# radvel derive -s $basepath
driver.derive(args)

# # optionally, make corner plot for derived parameters
# radvel plot -t derived -s $basepath
args.type = ['derived']
driver.plots(args)

# # do model comparison. valid choices: ['nplanets', 'e', 'trend', 'jit', 'gp']
# radvel ic -t nplanets e trend -s $basepath
args.type = ['nplanets', 'e', 'trend', 'jit']
args.mixed = True      # flag to compare all models with the fixed parameters mixed and matched rather than treating each model comparison separately. This is the default.
args.unmixed = False   # flag to treat each model comparison separately (without mixing them) rather than comparing all models with the fixed parameters mixed and matched.
args.fixjitter = False # flag to fix the stellar jitters at the nominal model best-fit value
args.verbose = True    # get more details

driver.ic_compare(args)

# # make the final report
# radvel report -s $basepath
driver.report(args)
