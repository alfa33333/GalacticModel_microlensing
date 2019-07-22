""" Defaul samplers """
import numpy as np
import dynesty
from dynesty import plotting as dyplot
import likelihoodfunctions_galmod as likemod
from galmoddefinitions import probte
from galmoddefinitions import thetae_func
import utils

class NestedSampling():
    """ This implementes a defaul nested sampling for
    the galactic model """
    def __init__(self, filesamples, nbins=20, column=6):
        self.xval, self.val = probte(filesamples, column, nbins)
        self.res = None

    def log_likelihood(self, sample):
        """ Simple wrapper for the probability """

        output = likemod.logprob_sourcenorm(sample, self.xval, self.val)
        return output

    def drun(self, dlogz=0.1, ndim=4):
        """ simple main function for sampling. """
        # initialize our nested sampler
        sampler = dynesty.NestedSampler(self.log_likelihood, likemod.pt_te, ndim)
        sampler.run_nested(dlogz=dlogz)
        self.res = sampler.results
        self.res.summary()
        fig, _ = dyplot.runplot(self.res, lnz_error=False)
        fig1, _ = dyplot.traceplot(self.res, truths=np.zeros(ndim), \
                                    truth_color='black', show_titles=True, \
                                    trace_cmap='viridis', connect=True, \
                                    connect_highlight=range(10))
        fig2, _ = dyplot.cornerplot(self.res, color='blue', \
                                truth_color='black', show_titles=True, \
                                max_n_ticks=3, quantiles=None)
        fig.savefig('./output/evidence.png')
        fig1.savefig('./output/tracers.png')
        fig2.savefig('./output/cornerplot.png')

################################################################
class NestedEmcee:
    """Class with a short Nested sampling followed by an Emcee run."""
    def __init__(self, filesamples, nbins=20, column=6, prefix='Emcee', defaul_labels=None):
        self.xval, self.val = probte(filesamples, column, nbins)
        self.res = None
        self.plotprefix = "./output/"+prefix
        if defaul_labels is None:
            self.labels = ['ogm', 'x', 'Ds', 'Vs']
        else:
            self.labels = defaul_labels
        self._nwalkers = 100
        self._nsteps = 100
        self._max_burnin_iterations = 20
        self._nsteps_production = 2000

    def _log_likelihood(self, sample):
        """ Simple wrapper for the log likelihood """

        output = likemod.logprob_sourcenorm(sample, self.xval, self.val)
        return output

    def logprob(self, variables):
        """ Simple Wrapper for the log probability"""
        if (-2 <= variables[0] <= 1.8) and  (0 < variables[1] <= 1) \
            and (0 < variables[2] <= 11000) and (0 < variables[3] <= 10):
            thetae = thetae_func(variables)
            if 8000 > thetae >= 72.27:
                output_log = likemod.logprob_sourcenorm(variables, self.xval, self.val)
                return output_log
            else:
                return -np.inf
        else:
            return -np.inf


    def _emcee_fit(self, parameters):
        import scipy.optimize as opt
        import emcee

        nwalkers = self._nwalkers
        nsteps = self._nsteps
        max_burnin_iterations = self._max_burnin_iterations
        nsteps_production = self._nsteps_production

        #print('Initial parameters:', parameters)
        print('ln Prob = ', self.logprob(parameters))

        ndim = len(parameters)

        print('Optimising...')

        neglogprob = lambda parameters: -self.logprob(parameters)
        res = opt.minimize(neglogprob, parameters, method='Nelder-Mead')

        parameters = res.x
        print('Optimized parameters:', parameters)
        print('ln Prob = ', self.logprob(parameters))

        #print('ndim, ',ndim, len(parameters), nwalkers)
        state = [parameters + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logprob)

        print("Running burn-in...")

        iteration = 0

        print('ndim, walkers, nsteps, max_iterations:', \
            ndim, nwalkers, nsteps, max_burnin_iterations)

        while iteration < max_burnin_iterations:

            state, _, _ = sampler.run_mcmc(state, nsteps)

            iteration += 1
            print('iteration', iteration, 'completed')

            kmax = np.argmax(sampler.flatlnprobability)
            parameters = sampler.flatchain[kmax, :]

            np.save(self.plotprefix+'-chain-burnin', sampler.flatchain)
            np.save(self.plotprefix+'-lnp-burnin', sampler.flatlnprobability)
            utils.plot_chain(sampler, ndim, self.labels, self.plotprefix, suffix='-burnin.png')

        print("Running production...")

        sampler.reset()

        state, _, _ = sampler.run_mcmc(state, nsteps_production)
        utils.plot_chain(sampler, ndim, self.labels, self.plotprefix, suffix='-production.png')

        np.save(self.plotprefix+'-chain-production', sampler.flatchain)
        np.save(self.plotprefix+'-lnp-production', sampler.flatlnprobability)
        np.save(self.plotprefix+'-state-production', np.asarray(state))
        np.save(self.plotprefix+'-min_chi2-production', \
            np.asarray(sampler.flatchain[np.argmax(sampler.flatlnprobability)]))

        print('Finished')

    def run(self, dlogz=10):
        """Main method to combine short nested sampling and Emcee"""

        ndim = len(self.labels)

        print('Initial Nested sampling')
        sampler = dynesty.NestedSampler(self._log_likelihood, likemod.pt_te, ndim, nlive=100)
        sampler.run_nested(dlogz=dlogz)
        res = sampler.results
        print("\nStarting Emcee")
        initial_value = res['samples'][-1]
        print("Initial values are:")
        print("{:s} = {:f}".format(self.labels[0], initial_value[0]))
        print("{:s} = {:f}".format(self.labels[1], initial_value[1]))
        print("{:s} = {:f}".format(self.labels[2], initial_value[2]))
        print("{:s} = {:f}".format(self.labels[3], initial_value[3]))
        self._emcee_fit(initial_value)
