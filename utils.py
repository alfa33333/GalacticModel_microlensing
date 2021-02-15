"""Module containing any utility function or routine not related with
the physics or sampling."""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pylab import subplots_adjust
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import corner as cr

def save_obj(obj, name):
    """
    Method to save pickle objects.
    """
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Method to load pickle objects.
    """
    with open(name + '.pkl', 'rb') as file:
        return pickle.load(file)

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

class Postprocessing():
    def __init__(self, bulge_path, disk_path):
        self.weightedouput_bulge = load_obj(bulge_path)
        self.weightedouput_disk = load_obj(disk_path)
        self.samplesbulge_equal = []
        self.samplesdisc_equal = []
        self.total_samples = []

    def full_postprocessing(self):
        self.basic_postprocessing()
        self.plot_total_weighted(self.total_samples,"Total")
        self.plot_total_weighted(self.samplesbulge_equal,"Bulge")
        self.plot_total_weighted(self.samplesdisc_equal,"Disk")
        self.plot_individual_weighted(self.samplesbulge_equal, self.samplesdisc_equal, self.total_samples)

    def basic_postprocessing(self):
        self.plot_unweighted(self.weightedouput_bulge, "bulge")
        self.plot_unweighted(self.weightedouput_disk, "disk")
        self.samplesbulge_equal, self.samplesdisc_equal = \
                                self.run_uniform_weight(self.weightedouput_bulge, \
                                                        self.weightedouput_disk)
        self.total_samples = np.vstack((self.samplesdisc_equal, self.samplesbulge_equal))

    @staticmethod
    def plot_unweighted(weightedouput, population):
        fig1, _ = dyplot.cornerplot(weightedouput, color='blue', \
                    truth_color='black', show_titles=True, max_n_ticks=3, quantiles=None)
        fig2, _ = dyplot.traceplot(weightedouput, truth_color='black', \
                     show_titles=True, trace_cmap='viridis')
        fig3, _ = dyplot.runplot(weightedouput, lnz_error=False)
        fig1.savefig("./output/"+population+"_corner_unweighted.png")
        fig2.savefig("./output/"+population+"_trace_unweighted.png")
        fig3.savefig("./output/"+population+"_runplot_unweighted.png")

    @staticmethod
    def plot_total_weighted(samples, population):
        figure = cr.corner(samples[:, :4], weights=samples[:, 4]/samples[:, 5], \
          labels=[r"$\log_{10}{M_L}$", r"$D_S$", r"$DL/DS$", r"$v/v_c$"], show_titles=True)
        figure.savefig("./output/Galmod_"+population+".png")

    @staticmethod
    def plot_individual_weighted(samplesbulge_equal, samplesdisc_equal, total_samples):
        _, edgest, _, totala, totalb, totalc = Postprocessing.samplingadjust(samplesbulge_equal, samplesdisc_equal, total_samples, 1)
        Postprocessing.plotadjust(samplesbulge_equal, samplesdisc_equal, \
        total_samples, 1, totala, totalb, totalc, edgest)

    @staticmethod
    def run_uniform_weight(weightedouput_bulge, weightedouput_disc):
        samplesbulge, weightsbulge = weightedouput_bulge.samples, \
            np.exp(weightedouput_bulge.logwt - weightedouput_bulge.logz[-1])
        samplesbulge_equal = dyfunc.resample_equal(samplesbulge, weightsbulge)
        samplesdisc, weightsdisc = weightedouput_disc.samples, \
            np.exp(weightedouput_disc.logwt - weightedouput_disc.logz[-1])
        samplesdisc_equal = dyfunc.resample_equal(samplesdisc, weightsdisc)
        return samplesbulge_equal, samplesdisc_equal

    @staticmethod
    def samplingadjust(samplesbulge_equal, samplesdisc_equal, total_samples, column):
        countst, edgest = np.histogram(total_samples[:, column],\
             weights=total_samples[:, 4]/total_samples[:, 5], bins=30)
        total_norm = np.sum((countst) * np.diff(edgest))
        totala = np.ones_like(total_samples[:, column])
        totalb = np.ones_like(samplesbulge_equal[:, column])
        totalc = np.ones_like(samplesdisc_equal[:, column])
        totala = total_norm*totala
        totalb = total_norm*totalb
        totalc = total_norm*totalc
        return countst, edgest, total_norm, totala, totalb, totalc

    @staticmethod
    def plotadjust(samplesbulge_equal, samplesdisc_equal, \
        total_samples, column, totala, totalb, totalc, edgest):

        fig, ax = plt.subplots()

        ax.hist(samplesbulge_equal[:, column], \
            weights=(samplesbulge_equal[:, 4]/samplesbulge_equal[:, 5])/totalb, bins=edgest, \
                color='r', histtype='step', fill=False, label='Bulge')
        ax.hist(samplesdisc_equal[:, column], \
            weights=(samplesdisc_equal[:, 4]/samplesdisc_equal[:, 5])/totalc, bins=edgest, \
                color='b', histtype='step', fill=False, label='Disc')
        ax.hist(total_samples[:, column], \
            weights=(total_samples[:, 4]/total_samples[:, 5])/totala, bins=edgest, \
                color='k', histtype='step', fill=False, label='Total')
        ax.legend()
        ax.set_xlabel("pc")
        fig.suptitle("DS")
        
        fig.savefig('./output/DSoutputc_te.png')
    
    @staticmethod
    def integralportion(samplesbulgeb_equal, samplesdiscb_equal, \
        total_samplesb, edgest, totala, totalb, totalc, column):

        countstx, edgestx = np.histogram(total_samplesb[:, column], \
            weights=(total_samplesb[:, 4]/total_samplesb[:, 5])/totala, bins=edgest)
        countsbx, edgesbx = np.histogram(samplesbulgeb_equal[:, column], \
            weights=(samplesbulgeb_equal[:, 4]/samplesbulgeb_equal[:, 5])/totalb, bins=edgest)
        countsdx, edgesdx = np.histogram(samplesdiscb_equal[:, column], \
            weights=(samplesdiscb_equal[:, 4]/samplesdiscb_equal[:, 5])/totalc, bins=edgest)
        print('Total ', np.sum((countstx) * np.diff(edgestx)))
        print('Bulge ', np.sum((countsbx) * np.diff(edgesbx)))
        print('Disc ', np.sum((countsdx) * np.diff(edgesdx)))

if __name__ == '__main__':
    postprocessing = Postprocessing("./output/weighted_bulge_te", "./output/weighted_disk_te")
    postprocessing.full_postprocessing()
