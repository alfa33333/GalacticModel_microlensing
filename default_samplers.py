""" Defaul samplers """
import numpy as np
import dynesty
from dynesty import plotting as dyplot
import likelihoodfunctions_galmod as likemod
from galmoddefinitions import probte

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
        fig.savefig('./evidence.png')
        fig1.savefig('./tracers.png')
        fig2.savefig('./cornerplot.png')
