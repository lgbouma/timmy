"""
env: py37_emcee2
"""

import radvel
from radvel import driver
from radvel.plot import orbit_plots, mcmc_plots
from radvel.mcmc import statevars
from radvel import plot
from radvel.utils import t_to_phase, fastbin, sigfig
from radvel.driver import load_status, save_status
import configparser, os, emcee

import numpy as np
import matplotlib
from matplotlib import rcParams, gridspec
from matplotlib import pyplot as plt
from matplotlib.cm import nipy_spectral
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from astropy.time import Time

from astropy import units as units, constants as const

##########
# config #
##########
latex = {
    'ms': r'm s$^{\mathregular{-1}}$',
    'BJDTDB': r'BJD$_{\mathregular{TDB}}$'
}

telfmts_default = {
    'j': dict(color='C2', marker=u'o', label='HIRES', mew=1),
    'k': dict(color='k', fmt='s', mfc='none', label='HIRES pre 2004', mew=1),
    'a': dict(color='g', fmt='d', label='APF'),
    'pfs': dict(color='magenta', fmt='p', label='PFS'),
    'CORALIE': dict(color='C0', fmt='d', label='CORALIE'),
    'h': dict(color='C1', fmt="s", label='HARPS'),
    'harps-n': dict(color='firebrick', fmt='^', label='HARPS-N'),
    'l': dict(color='g', fmt='*', label='LICK'),
}
telfmts_default['lick'] = telfmts_default['l']
telfmts_default['hires_rj'] = telfmts_default['j']
telfmts_default['hires'] = telfmts_default['j']
telfmts_default['hires_rk'] = telfmts_default['k']
telfmts_default['apf'] = telfmts_default['a']
telfmts_default['harps'] = telfmts_default['h']
telfmts_default['LICK'] = telfmts_default['l']
telfmts_default['HIRES_RJ'] = telfmts_default['j']
telfmts_default['HIRES'] = telfmts_default['j']
telfmts_default['HIRES_RK'] = telfmts_default['k']
telfmts_default['APF'] = telfmts_default['a']
telfmts_default['HARPS'] = telfmts_default['h']
telfmts_default['HARPS-N'] = telfmts_default['harps-n']
telfmts_default['PFS'] = telfmts_default['pfs']


cmap = nipy_spectral
default_colors = ['orange', 'purple', 'magenta', 'pink', 'green', 'grey', 'red']
from aesthetic.plot import set_style
set_style()

highlight_format = dict(marker='o', ms=16, mfc='none', mew=2, mec='gold', zorder=99)


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


def limit_plots(args):
    """
    Generate plots

    Args:
        args (ArgumentParser): command line arguments
    """

    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]
    statfile = os.path.join(
        args.outputdir, "{}_radvel.stat".format(conf_base)
    )

    status = load_status(statfile)

    assert status.getboolean('fit', 'run'), \
        "Must perform max-liklihood fit before plotting"

    postpath = status.get('fit', 'postfile')
    postpath = os.path.join(
        args.outputdir, os.path.basename(postpath)
    )
    if not os.path.exists(postpath):
        raise FileNotFoundError(f'expected posterior file to exist, {post}')

    post = radvel.posterior.load(postpath)

    # from timmy.driver.soneoff_drivers, 99.7th percetile
    logk1_limit = 4.68726682
    post.params['logk1'] = radvel.Parameter(value=logk1_limit)
    post.params['k1'] = radvel.Parameter(value=np.exp(logk1_limit))

    for ptype in args.type:
        print("Creating {} plot for {}".format(ptype, conf_base))

        if ptype == 'rv':
            args.plotkw['uparams'] = post.uparams
            saveto = os.path.join(
                args.outputdir,conf_base+'_rvlimit_multipanel.pdf'
            )
            P, _ = radvel.utils.initialize_posterior(config_file)
            if hasattr(P, 'bjd0'):
                args.plotkw['epoch'] = P.bjd0

            # import IPython; IPython.embed()
            # P.params (set logk1 to whatever...)
            # assert 0

            RVPlot = MultipanelPlot(
                post, saveplot=saveto, **args.plotkw
            )
            RVPlot.plot_multipanel()

        savestate = {'{}_plot'.format(ptype): os.path.relpath(saveto)}
        save_status(statfile, 'plot', savestate)


class MultipanelPlot(object):
    """
    Class to handle the creation of RV multipanel plots.

    Args:
        post (radvel.Posterior): radvel.Posterior object. The model
            plotted will be generated from `post.params`
        epoch (int, optional): epoch to subtract off of all time measurements
        yscale_auto (bool, optional): Use matplotlib auto y-axis
             scaling (default: False)
        yscale_sigma (float, optional): Scale y-axis limits for all panels to be +/-
             yscale_sigma*(RMS of data plotted) if yscale_auto==False
        phase_nrows (int, optional): number of columns in the phase
            folded plots. Default is nplanets.
        phase_ncols (int, optional): number of columns in the phase
            folded plots. Default is 1.
        uparams (dict, optional): parameter uncertainties, must
           contain 'per', 'k', and 'e' keys.
        telfmts (dict, optional): dictionary of dictionaries mapping
            instrument suffix to plotting format code. Example:
                telfmts = {
                     'hires': dict(fmt='o',label='HIRES'),
                     'harps-n' dict(fmt='s')
                }
        legend (bool, optional): include legend on plot? Default: True.
        phase_limits (list, optional): two element list specifying
            pyplot.xlim bounds for phase-folded plots. Useful for
            partial orbits.
        nobin (bool, optional): If True do not show binned data on
            phase plots. Will default to True if total number of
            measurements is less then 20.
        phasetext_size (string, optional): fontsize for text in phase plots.
            Choice of {'xx-small', 'x-small', 'small', 'medium', 'large',
            'x-large', 'xx-large'}. Default: 'x-small'.
        rv_phase_space (float, optional): amount of space to leave between orbit/residual plot
            and phase plots.
        figwidth (float, optional): width of the figures to be produced.
            Default: 7.5 (spans a page with 0.5 in margins)
        fit_linewidth (float, optional): linewidth to use for orbit model lines in phase-folded
            plots and residuals plots.
        set_xlim (list of float): limits to use for x-axes of the timeseries and residuals plots, in
            JD - `epoch`. Ex: [7000., 70005.]
        text_size (int): set matplotlib.rcParams['font.size'] (default: 9)
        highlight_last (bool): make the most recent measurement much larger in all panels
        show_rms (bool): show RMS of the residuals by instrument in the legend
        legend_kwargs (dict): dict of options to pass to legend (plotted in top panel)
    """
    def __init__(self, post, saveplot=None, epoch=2450000, yscale_auto=False,
                 yscale_sigma=3.0, phase_nrows=None, phase_ncols=None,
                 uparams=None, telfmts={}, legend=True, phase_limits=[],
                 nobin=False, phasetext_size='medium', rv_phase_space=0.08,
                 figwidth=4.2, fit_linewidth=1.0, set_xlim=None, text_size=11,
                 highlight_last=False, show_rms=False,
                 legend_kwargs=dict(loc='best')):

        self.post = post
        self.saveplot = saveplot
        self.epoch = epoch
        self.yscale_auto = yscale_auto
        self.yscale_sigma = yscale_sigma
        if phase_ncols is None:
            self.phase_ncols = 1
        if phase_nrows is None:
            self.phase_nrows = self.post.likelihood.model.num_planets
        self.uparams = uparams
        self.rv_phase_space = rv_phase_space
        self.telfmts = telfmts
        self.legend = legend
        self.phase_limits = phase_limits
        self.nobin = nobin
        self.phasetext_size = phasetext_size
        self.figwidth = figwidth
        self.fit_linewidth = fit_linewidth
        self.set_xlim = set_xlim
        self.highlight_last = highlight_last
        self.show_rms = show_rms
        self.legend_kwargs = legend_kwargs
        rcParams['font.size'] = text_size

        if isinstance(self.post.likelihood, radvel.likelihood.CompositeLikelihood):
            self.like_list = self.post.likelihood.like_list
        else:
            self.like_list = [self.post.likelihood]

        # FIGURE PROVISIONING
        # self.ax_rv_height = self.figwidth * 0.6
        # self.ax_phase_height = self.ax_rv_height / 1.4
        self.ax_rv_height = self.figwidth * 0.5
        self.ax_phase_height = self.ax_rv_height

        # convert params to synth basis
        synthparams = self.post.params.basis.to_synth(self.post.params)
        self.post.params.update(synthparams)

        self.model = self.post.likelihood.model
        self.rvtimes = self.post.likelihood.x
        self.rverr = self.post.likelihood.errorbars()
        self.num_planets = self.model.num_planets

        self.rawresid = self.post.likelihood.residuals()

        self.resid = (
            self.rawresid + self.post.params['dvdt'].value*(self.rvtimes-self.model.time_base)
            + self.post.params['curv'].value*(self.rvtimes-self.model.time_base)**2
        )

        if self.saveplot is not None:
            resolution = 10000
        else:
            resolution = 2000

        periods = []
        for i in range(self.num_planets):
            periods.append(synthparams['per%d' % (i+1)].value)
        if len(periods) > 0:
            longp = max(periods)
        else:
            longp = max(self.post.likelihood.x) - min(self.post.likelihood.x)

        self.dt = max(self.rvtimes) - min(self.rvtimes)
        self.rvmodt = np.linspace(
            min(self.rvtimes) - 0.05 * self.dt, max(self.rvtimes) + 0.05 * self.dt + longp,
            int(resolution)
        )

        self.orbit_model = self.model(self.rvmodt)
        self.rvmod = self.model(self.rvtimes)

        if ((self.rvtimes - self.epoch) < -2.4e6).any():
            self.plttimes = self.rvtimes
            self.mplttimes = self.rvmodt
        elif self.epoch == 0:
            self.epoch = 2450000
            self.plttimes = self.rvtimes - self.epoch
            self.mplttimes = self.rvmodt - self.epoch
        else:
            self.plttimes = self.rvtimes - self.epoch
            self.mplttimes = self.rvmodt - self.epoch

        self.slope = (
            self.post.params['dvdt'].value * (self.rvmodt-self.model.time_base)
            + self.post.params['curv'].value * (self.rvmodt-self.model.time_base)**2
        )
        self.slope_low = (
            self.post.params['dvdt'].value * (self.rvtimes-self.model.time_base)
            + self.post.params['curv'].value * (self.rvtimes-self.model.time_base)**2
        )

        # list for Axes objects
        self.ax_list = []

    def plot_timeseries(self, ylim=(-510, 510)):
        """
        Make a plot of the RV data and model in the current Axes.
        """

        ax = plt.gca()

        ax.axhline(0, color='k', linestyle=':', linewidth=1)

        if self.show_rms:
            rms_values = dict()
            for like in self.like_list:
                inst = like.suffix
                rms = np.std(like.residuals())
                rms_values[inst] = rms
        else:
            rms_values = False

        # plot orbit model
        # ax.plot(self.mplttimes, self.orbit_model, 'b-', rasterized=False, lw=self.fit_linewidth)

        # plot data
        vels = self.rawresid+self.rvmod
        mtelplot(
            # data = residuals + model
            self.plttimes, vels, self.rverr, self.post.likelihood.telvec, ax, telfmts=self.telfmts,
            rms_values=rms_values
        )

        if self.set_xlim is not None:
            ax.set_xlim(self.set_xlim)
        else:
            ax.set_xlim(min(self.plttimes)-0.01*self.dt, max(self.plttimes)+0.01*self.dt)
        plt.setp(ax.get_xticklabels(), visible=False)

        if self.highlight_last:
            ind = np.argmax(self.plttimes)
            plt.plot(self.plttimes[ind], vels[ind], **plot.highlight_format)

        # legend
        if self.legend:
            ax.legend(numpoints=1, **self.legend_kwargs)

        # years on upper axis
        axyrs = ax.twiny()
        xl = np.array(list(ax.get_xlim())) + self.epoch
        decimalyear = Time(xl, format='jd', scale='utc').decimalyear
        axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
        axyrs.set_xlim(*decimalyear)
        axyrs.set_xlabel('Year')
        plt.locator_params(axis='x', nbins=5)

        # if not self.yscale_auto:
        #     scale = np.std(self.rawresid+self.rvmod)
        #     ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)

        if isinstance(ylim, tuple):
            ax.set_ylim(ylim)

        ax.set_ylabel('RV [{ms:}]'.format(**plot.latex))
        ax.set_xlabel('Time [JD - {:d}]'.format(int(np.round(self.epoch))))
        # ticks = ax.yaxis.get_majorticklocs()
        # ax.yaxis.set_ticks(ticks[1:])

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.tick_params(right=True, which='both', direction='in')
        axyrs.get_xaxis().set_tick_params(which='both', direction='in')

    def plot_residuals(self):
        """
        Make a plot of residuals and RV trend in the current Axes.
        """

        ax = plt.gca()

        ax.plot(self.mplttimes, self.slope, 'k-', lw=self.fit_linewidth)

        mtelplot(self.plttimes, self.resid, self.rverr, self.post.likelihood.telvec, ax, telfmts=self.telfmts)
        if not self.yscale_auto:
            scale = np.std(self.resid)
            ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)

        if self.highlight_last:
            ind = np.argmax(self.plttimes)
            plt.plot(self.plttimes[ind], self.resid[ind], **plot.highlight_format)

        if self.set_xlim is not None:
            ax.set_xlim(self.set_xlim)
        else:
            ax.set_xlim(min(self.plttimes)-0.01*self.dt, max(self.plttimes)+0.01*self.dt)

        ##########################################
        # what would explain the Pdot from transits?
        period = 1.338231466*units.day
        Pdot_tra = -2.736e-10
        Pdot_err = 2**(1/2.)*2.83e-11 # inflating appropriately
        Pdot_tra_perr = Pdot_tra + Pdot_err
        Pdot_tra_merr = Pdot_tra - Pdot_err
        dvdt_tra = (Pdot_tra * const.c / period).to(
            (units.m/units.s)/units.day).value
        dvdt_tra_perr = (Pdot_tra_perr * const.c / period).to(
            (units.m/units.s)/units.day).value
        dvdt_tra_merr = (Pdot_tra_merr * const.c / period).to(
            (units.m/units.s)/units.day).value

        # model times are now an arrow band
        _mtimes = np.linspace(np.min(self.plttimes)+3500,
                              np.min(self.plttimes)+4000, num=2000)
        _mbase = np.nanmedian(_mtimes)
        model_tra_line = dvdt_tra*(_mtimes-_mbase)
        model_tra_merr = dvdt_tra_merr*(_mtimes-_mbase)# + curv*(_times-time_base)**2
        model_tra_perr = dvdt_tra_perr*(_mtimes-_mbase)# + curv*(_times-time_base)**2

        yoffset = 35
        ax.plot(_mtimes, model_tra_line+yoffset,
                color='purple', zorder=-3, lw=0.5, ls='-', linewidth=self.fit_linewidth)
        #ax.fill_between(_mtimes, model_tra_merr+yoffset, model_tra_perr+yoffset,
        #                color='purple', zorder=-4, alpha=0.9, lw=0)
        ax.text(0.92, 0.85, 'Slope = $c\dot{P}/P$', va='top', ha='right',
                transform=ax.transAxes, color='purple', alpha=0.9,
                fontsize='large')

        ##########################################

        ticks = ax.yaxis.get_majorticklocs()
        ax.yaxis.set_ticks([ticks[0], 0.0, ticks[-1]])
        plt.xlabel('Time [JD - {:d}]'.format(int(np.round(self.epoch))))
        ax.set_ylabel('Residuals [{ms:}]'.format(**plot.latex))
        ax.yaxis.set_major_locator(MaxNLocator(5, prune='both'))

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.tick_params(right=True, which='both', direction='in')
        ax.tick_params(top=True, which='both', direction='in')


    def plot_phasefold(self, pltletter, pnum):
        """
        Plot phased orbit plots for each planet in the fit.

        Args:
            pltletter (int): integer representation of
                letter to be printed in the corner of the first
                phase plot.
                Ex: ord("a") gives 97, so the input should be 97.
            pnum (int): the number of the planet to be plotted. Must be
                the same as the number used to define a planet's
                Parameter objects (e.g. 'per1' is for planet #1)

        """

        ax = plt.gca()

        if len(self.post.likelihood.x) < 20:
            self.nobin = True

        bin_fac = 1.75
        bin_markersize = bin_fac * rcParams['lines.markersize']
        bin_markeredgewidth = bin_fac * rcParams['lines.markeredgewidth']

        rvmod2 = self.model(self.rvmodt, planet_num=pnum) - self.slope
        modph = t_to_phase(self.post.params, self.rvmodt, pnum, cat=True) - 1
        rvdat = self.rawresid + self.model(self.rvtimes, planet_num=pnum) - self.slope_low
        phase = t_to_phase(self.post.params, self.rvtimes, pnum, cat=True) - 1
        rvdatcat = np.concatenate((rvdat, rvdat))
        rverrcat = np.concatenate((self.rverr, self.rverr))
        rvmod2cat = np.concatenate((rvmod2, rvmod2))
        bint, bindat, binerr = fastbin(phase+1, rvdatcat, nbins=25)
        bint -= 1.0

        ax.axhline(0, color='k', linestyle=':', linewidth=1)
        ax.plot(sorted(modph), rvmod2cat[np.argsort(modph)], 'k--', linewidth=self.fit_linewidth)
        #plot.labelfig(pltletter)

        telcat = np.concatenate((self.post.likelihood.telvec, self.post.likelihood.telvec))

        if self.highlight_last:
            ind = np.argmax(self.rvtimes)
            hphase = t_to_phase(self.post.params, self.rvtimes[ind], pnum, cat=False)
            if hphase > 0.5:
                hphase -= 1
            plt.plot(hphase, rvdatcat[ind], **plot.highlight_format)

        mtelplot(phase, rvdatcat, rverrcat, telcat, ax, telfmts=self.telfmts)
        if not self.nobin and len(rvdat) > 10:
            pass
            #ax.errorbar(
            #    bint, bindat, yerr=binerr, fmt='ro', mec='w', ms=bin_markersize,
            #    mew=bin_markeredgewidth
            #)

        if self.phase_limits:
            ax.set_xlim(self.phase_limits[0], self.phase_limits[1])
        else:
            ax.set_xlim(-0.5, 0.5)

        if not self.yscale_auto:
            scale = np.std(rvdatcat)
            ax.set_ylim(-self.yscale_sigma*scale, self.yscale_sigma*scale)
        ax.set_ylim((-510, 510))

        keys = [p+str(pnum) for p in ['per', 'k', 'e']]

        labels = [self.post.params.tex_labels().get(k, k) for k in keys]
        if pnum < self.num_planets:
            ticks = ax.yaxis.get_majorticklocs()
            ax.yaxis.set_ticks(ticks[1:-1])

        ax.set_ylabel('RV [{ms:}]'.format(**plot.latex))
        ax.set_xlabel('Phase')

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.tick_params(right=True, which='both', direction='in')
        ax.tick_params(top=True, which='both', direction='in')

        print_params = ['per', 'k', 'e']
        units = {'per': 'days', 'k': plot.latex['ms'], 'e': ''}

        anotext = []
        for l, p in enumerate(print_params):
            val = self.post.params["%s%d" % (print_params[l], pnum)].value

            if self.uparams is None:
                _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])
            else:
                if hasattr(self.post, 'medparams'):
                    val = self.post.medparams["%s%d" % (print_params[l], pnum)]
                else:
                    print("WARNING: medparams attribute not found in " +
                          "posterior object will annotate with " +
                          "max-likelihood values and reported uncertainties " +
                          "may not be appropriate.")
                err = self.uparams["%s%d" % (print_params[l], pnum)]
                if err > 1e-15:
                    val, err, errlow = sigfig(val, err)
                    _anotext = '$\\mathregular{%s}$ = %s $\\mathregular{\\pm}$ %s %s' \
                               % (labels[l].replace("$", ""), val, err, units[p])
                else:
                    _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])

            anotext += [_anotext]

        #anotext = '\n'.join(anotext)
        anotext = anotext[1] # just the semi-amplitude
        logk1_limit = 4.68726682
        #anotext = (
        #    f'K < {np.exp(logk1_limit):.1f}' +' m$\,$s$^{-1}$ (3$\sigma$)\n'
        #    '$M_{\mathrm{p}} \sin i < 1.20\,M_{\mathrm{Jup}}$'
        #)
        anotext = (
            #f'K < {np.exp(logk1_limit):.1f}' +' m$\,$s$^{-1}$ (3$\sigma$)\n'
            '$M_{\mathrm{p}} \sin i < 1.20\,M_{\mathrm{Jup}}\ (3\sigma)$'
        )

        add_anchored(
            anotext, loc='lower left', frameon=True, prop=dict(size=self.phasetext_size),
            bbox=dict(ec='none', fc='w', alpha=0.8)
        )


    def plot_multipanel(self, nophase=False, letter_labels=False):
        """
        Provision and plot an RV multipanel plot

        Args:
            nophase (bool, optional): if True, don't
                include phase plots. Default: False.
            letter_labels (bool, optional): if True, include
                letter labels on orbit and residual plots.
                Default: True.

        Returns:
            tuple containing:
                - current matplotlib Figure object
                - list of Axes objects
        """

        if nophase:
            scalefactor = 1
        else:
            scalefactor = self.phase_nrows

        figheight = self.ax_rv_height + self.ax_phase_height * scalefactor

        # provision figure
        fig = plt.figure(figsize=(self.figwidth, figheight+1.0))

        fig.subplots_adjust(left=0.12, right=0.95)
        gs_rv = gridspec.GridSpec(1, 1, height_ratios=[1.])

        divide = 1 - self.ax_rv_height / figheight
        gs_rv.update(left=0.12, right=0.93, top=0.93,
                     bottom=divide+self.rv_phase_space*0.5, hspace=0.)

        # orbit plot
        ax_rv = plt.subplot(gs_rv[0, 0])
        self.ax_list += [ax_rv]

        plt.sca(ax_rv)
        self.plot_timeseries()
        pltletter = ord('a')
        if letter_labels:
            plot.labelfig(pltletter)
            pltletter += 1

        # # residuals
        # ax_resid = plt.subplot(gs_rv[1, 0])
        # self.ax_list += [ax_resid]

        # plt.sca(ax_resid)
        # self.plot_residuals()
        # if letter_labels:
        #     plot.labelfig(pltletter)
        #     pltletter += 1

        # phase-folded plots
        if not nophase:
            gs_phase = gridspec.GridSpec(self.phase_nrows, self.phase_ncols)

            if self.phase_ncols == 1:
                gs_phase.update(left=0.12, right=0.93,
                                top=divide - self.rv_phase_space * 0.5,
                                bottom=0.07, hspace=0.003)
            else:
                gs_phase.update(left=0.12, right=0.93,
                                top=divide - self.rv_phase_space * 0.5,
                                bottom=0.07, hspace=0.25, wspace=0.25)

            for i in range(self.num_planets):
                i_row = int(i / self.phase_ncols)
                i_col = int(i - i_row * self.phase_ncols)
                ax_phase = plt.subplot(gs_phase[i_row, i_col])
                self.ax_list += [ax_phase]

                plt.sca(ax_phase)
                self.plot_phasefold(pltletter, i+1)
                pltletter += 1

        if self.saveplot is not None:
            fig.tight_layout(w_pad=2, h_pad=2)
            plt.savefig(self.saveplot, dpi=150, bbox_inches='tight')
            print("RV multi-panel plot saved to %s" % self.saveplot)

        return fig, self.ax_list


def telplot(x, y, e, tel, ax, lw=1., telfmt={}, rms=0):
    """Plot data from from a single telescope

    x (array): Either time or phase
    y (array): RV
    e (array): RV error
    tel (string): telecsope string key
    ax (matplotlib.axes.Axes): current Axes object
    lw (float): line-width for error bars
    telfmt (dict): dictionary corresponding to kwargs
        passed to errorbar. Example:

        telfmt = dict(fmt='o',label='HIRES',color='red')
    """

    # Default formatting
    kw = dict(
        fmt='o', capsize=0, mew=0,
        ecolor='0.6', lw=lw, color='orange',
    )

    # If not explicit format set, look among default formats
    if not telfmt and tel in telfmts_default:
        telfmt = telfmts_default[tel]

    for k in telfmt:
        kw[k] = telfmt[k]

    if not 'label' in kw.keys():
        if tel in telfmts_default:
            kw['label'] = telfmts_default[tel]['label']
        else:
            kw['label'] = tel

    if rms:
        kw['label'] += '\nRMS={:.2f} {:s}'.format(rms, latex['ms'])

    plt.errorbar(x, y, yerr=e, **kw)


def mtelplot(x, y, e, tel, ax, lw=1., telfmts={}, **kwargs):
    """
    Overplot data from from multiple telescopes.

    x (array): Either time or phase
    y (array): RV
    e (array): RV error
    tel (array): array of telecsope string keys
    ax (matplotlib.axes.Axes): current Axes object
    telfmts (dict): dictionary of dictionaries corresponding to kwargs
        passed to errorbar. Example:

        telfmts = {
             'hires': dict(fmt='o',label='HIRES'),
             'harps-n' dict(fmt='s')
        }
    """

    rms_values = kwargs.pop('rms_values', False)

    utel = np.unique(tel)

    ci = 0
    for t in utel:
        xt = x[tel == t]
        yt = y[tel == t]
        et = e[tel == t]

        telfmt = {}

        if t in telfmts:
            telfmt = telfmts[t]
            if 'color' not in telfmt:
                telfmt['color'] = default_colors[ci]
                ci +=1
        elif t not in telfmts and t not in telfmts_default:
            telfmt = dict(color=default_colors[ci])
            ci +=1
        else:
            telfmt = {}

        if rms_values:
            rms = rms_values[t]
        else:
            rms = 0

        telplot(xt, yt, et, t, ax, lw=1., telfmt=telfmt, rms=rms)

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useOffset=False)
    )


def add_anchored(*args, **kwargs):
    """
    Add text at a particular location in the current Axes

    Args:
        s (string): text
        loc (string): location code
        pad (float [optional]): pad between the text and the frame
            as fraction of the font size
        borderpad (float [optional]): pad between the frame and the axes (or *bbox_to_anchor*)
        prop (matplotlib.font_manager.FontProperties): font properties
    """

    bbox = {}
    if 'bbox' in kwargs:
        bbox = kwargs.pop('bbox')
    at = AnchoredText(*args, **kwargs)
    if len(bbox.keys()) > 0:
        plt.setp(at.patch, **bbox)

    ax = plt.gca()
    ax.add_artist(at)


def plot_rvs():

    setupfn = "/home/luke/Dropbox/proj/timmy/drivers/radvel_drivers/TOI837.py"
    outputdir = "/home/luke/Dropbox/proj/timmy/results/radvel_fitting/20200624_simple_planet"

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    args = args_object(setupfn, outputdir)

    # plot the upper limit (99.7th pctile) fit, which has already been
    # performed in 20200624_simple_planet
    args.type = ['rv']
    limit_plots(args) # pulled from radvel.driver.plots


if __name__ == "__main__":
    plot_rvs()
