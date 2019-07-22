"""Module containing any utility function or routine not related with
the physics or sampling."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pylab import subplots_adjust


def plot_chain(sampler, ndim, parameter_labels, plotprefix, plot_lnprob=True, suffix=''):
    """Utility module: It plots the chains from a montecarlo process."""

    index = list(range(ndim))

    labels = parameter_labels

    plt.figure(figsize=(8, 11))

    subplots_adjust(hspace=0.0001)

    for i in range(ndim):

        if i == 0:
            plt.subplot(ndim+plot_lnprob, 1, i+1)
            ax1 = plt.gca()
        else:
            plt.subplot(ndim+plot_lnprob, 1, i+1, sharex=ax1)

        plt.plot(sampler.chain[:, :, index[i]].T, '-', color='k', alpha=0.3)

        if labels:
            plt.ylabel(labels[i])

        axis = plt.gca()

        if i < ndim-1+plot_lnprob:
            plt.setp(axis.get_xticklabels(), visible=False)
            axis.yaxis.set_major_locator(MaxNLocator(prune='lower'))
            axis.locator_params(axis='y', nbins=4)

    if plot_lnprob:
        plt.subplot(ndim+plot_lnprob, 1, ndim+plot_lnprob, sharex=ax1)
        plt.plot(sampler.lnprobability.T, '-', color='r', alpha=0.3)
        plt.ylabel(r"$ln P$")
        axis = plt.gca()
        axis.yaxis.set_major_locator(MaxNLocator(prune='lower'))
        axis.locator_params(axis='y', nbins=4)

    plt.savefig(plotprefix+suffix)
    plt.close()
    