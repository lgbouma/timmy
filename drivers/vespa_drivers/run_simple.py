#!/usr/bin/env python
"""
Hack of VESPA-0.6 `calcfpp` script for incorporating observational constraints
into both the FPP calculation, and also the quicklook summary plots.

Procedure to run this is:
    $ calcfpp -n 20000 dirname_with_ini_scripts
    $ python run_simple dirname_with_ini_scripts
"""
from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg')

import sys, os, re, time, os.path, glob
import argparse
import logging
import numpy as np
from six import string_types

from vespa import FPPCalculation
from vespa.stars.contrastcurve import ContrastCurveFromFile

from configobj import ConfigObj

import warnings
warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)

import logging

#utility function to initialize logging
def initLogging(filename, logger):
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(sh)
    return logger

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate FPP for a transit signal')

    parser.add_argument('folders', nargs='*', default=['.'],
                        help='Directory (or directories) for which to calculate FPP.  ' +
                        '`starfit --all <directory>` must have been run already for each directory.')
    parser.add_argument('-n','--n', type=int, default=20000,
                        help='Size of simulated populations (default=20000).')
    parser.add_argument('--recalc', action='store_true',
                        help='Delete all existing population simulations and recompute.')
    parser.add_argument('--debug', action='store_true',
                        help='Set logging level to DEBUG.')
    parser.add_argument('--newlog', action='store_true',
                        help='Start a fresh fpp.log file.')
    parser.add_argument('--recalc_lhood', action='store_true',
                        help='Recaclulate likelihoods instead of reading from cache file.')
    parser.add_argument('--refit_trsig', action='store_true',
                        help='Redo MCMC fit trapezoidal model to transit signal.')
    parser.add_argument('--bootstrap', type=int, default=0,
                        help='Number of bootstrap resamplings to do to estimate uncertainty of FPP. ' +
                             'Set to >1 to do bootstrap error estimates.')

    args = parser.parse_args()

    logger = None #dummy

    for folder in args.folders:
        #initialize logger for this folder:

        logfile = os.path.join(folder, 'calcfpp.log')
        if args.newlog:
            os.remove(logfile)
        logger = initLogging(logfile, logger)
        if args.debug:
            logger.setLevel(logging.DEBUG)

        try:
            do_from_ini = 0
            do_load = 1
            do_manual = 0

            if do_load:

                # initialize the transit signal and populations from the first
                # run-through of `calcfpp`.
                f = FPPCalculation.load(folder)
                ini_file = 'fpp.ini'

                config = ConfigObj(os.path.join(folder, ini_file))

                # Exclusion radius
                maxrad = float(config['constraints']['maxrad'])
                f.set_maxrad(maxrad)
                if 'secthresh' in config['constraints']:
                    secthresh = float(config['constraints']['secthresh'])
                    if not np.isnan(secthresh):
                        f.apply_secthresh(secthresh)

                # Odd-even constraint
                diff = 3 * np.max(f.trsig.depthfit[1])
                f.constrain_oddeven(diff)

                #apply contrast curve constraints if present
                if 'ccfiles' in config['constraints']:
                    ccfiles = config['constraints']['ccfiles']
                    if isinstance(ccfiles, string_types):
                        ccfiles = [ccfiles]
                    for ccfile in ccfiles:
                        if not os.path.isabs(ccfile):
                            ccfile = os.path.join(folder, ccfile)
                        m = re.search('(\w+)_(\w+)\.cc',os.path.basename(ccfile))
                        if not m:
                            logging.warning('Invalid CC filename ({}); '.format(ccfile) +
                                             'skipping.')
                            continue
                        else:
                            band = m.group(2)
                            inst = m.group(1)
                            name = '{} {}-band'.format(inst, band)
                            cc = ContrastCurveFromFile(ccfile, band, name=name)
                            f.apply_cc(cc)

                    # EB population notes:
                    # 'BEBs', 'BEBs (Double Period)', 'EBs', 'EBs (Double Period)', 'HEBs'

                    print(42*'=')
                    print('Original FPP')
                    print('FPP for {}: {}'.
                          format(f.name, f.FPP()))

                    skipmodels = ['EBs', 'EBs (Double Period)', 'HEBs',
                                  'HEBs (Double Period)']
                    print(42*'=')
                    print('Excluding {}'.format(repr(skipmodels)))
                    print('FPP for {}: {}'.
                          format(f.name, f.FPP(skipmodels=skipmodels)))

                    skipmodels = ['EBs', 'EBs (Double Period)']
                    print(42*'=')
                    print('Excluding {}'.format(repr(skipmodels)))
                    print('FPP for {}: {}'.
                          format(f.name, f.FPP(skipmodels=skipmodels)))


            if do_manual:

                raise NotImplementedError

                trsig_file = os.path.join(folder, 'trsig.pkl')
                if os.path.exists(trsig_file):
                    logging.info('Loading transit signal from {}...'.format(trsig_file))
                    with open(trsig_file, 'rb') as f:
                        trsig = pickle.load(f)

                f = FPPCalculation(trsig, popset, folder=folder)

            elif do_from_ini:
                # default from calcfpp
                f = FPPCalculation.from_ini(folder, ini_file='fpp.ini',
                                            recalc=args.recalc,
                                            ichrone='mist',
                                            n=args.n)
                # NOTE: this also applies contrast curves, etc.

            if args.refit_trsig:
                f.trsig.MCMC(refit=True)
                f.trsig.save(os.path.join(folder,'trsig.pkl'))

            trap_corner_file = os.path.join(folder, 'trap_corner.png')
            if not os.path.exists(trap_corner_file) or args.refit_trsig:
                f.trsig.corner(outfile=trap_corner_file)

            f.FPPplots(recalc_lhood=args.recalc_lhood)

            skipmodels = ['EBs', 'EBs (Double Period)', 'HEBs',
                          'HEBs (Double Period)']
            f.FPPhack(folder=folder, saveplot=True, figformat='png',
                      tag='no_EB_no_HEB',
                      skipmodels=skipmodels)

            skipmodels = ['EBs', 'EBs (Double Period)']
            f.FPPhack(folder=folder, saveplot=True, figformat='png',
                      tag='no_EB', skipmodels=skipmodels)


            if args.bootstrap > 0:
                logger.info('Re-fitting trapezoid MCMC model...')
                f.bootstrap_FPP(args.bootstrap)

            if args.bootstrap > 0:
                logger.info('Bootstrap results ({}) written to {}.'.
                            format(args.bootstrap,
                                   os.path.join(os.path.abspath(folder),
                                                'results_bootstrap.txt')))
                logger.info('FPP calculation successful. ' +
                            'Results/plots written to {}.'.format(os.path.abspath(folder)))

            print('FPP for {}: {}'.format(f.name,f.FPP()))

        except KeyboardInterrupt:
            raise

        except:
            logger.error('FPP calculation failed for {}.'.format(folder), exc_info=True)
