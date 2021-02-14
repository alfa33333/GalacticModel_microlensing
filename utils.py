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

    def basic_postprocessing(self):
        self.plot_unweighted(self.weightedouput_bulge, "bulge")
        self.plot_unweighted(self.weightedouput_disk, "disk")
        self.samplesbulge_equal, self.samplesdisc_equal = self.run_uniform_weight(self.weightedouput_bulge, self.weightedouput_disk)
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
    def run_uniform_weight(weightedouput_bulge, weightedouput_disc):
        samplesbulge, weightsbulge = weightedouput_bulge.samples, np.exp(weightedouput_bulge.logwt - weightedouput_bulge.logz[-1])
        samplesbulge_equal = dyfunc.resample_equal(samplesbulge, weightsbulge)
        samplesdisc, weightsdisc = weightedouput_disc.samples, np.exp(weightedouput_disc.logwt - weightedouput_disc.logz[-1])
        samplesdisc_equal = dyfunc.resample_equal(samplesdisc, weightsdisc)
        return samplesbulge_equal, samplesdisc_equal

    @staticmethod
    def samplingadjust(samplesbulge_equal, samplesdisc_equal, Total_samples, column):
        countsT, edgesT = np.histogram(Total_samples[:,column],weights=Total_samples[:,4]/Total_samples[:,5],bins=30)
        # countsTB, edgesTB = np.histogram(san_bulge_equal[:,2],weights=san_bulge_equal[:,4],bins=edgesT)
        # countsTD, edgesTD = np.histogram(san_disc_equal[:,2],weights=san_disc_equal[:,2],bins=edgesT)
        Total_norm = np.sum((countsT) * np.diff(edgesT))
        Totala = np.ones_like(Total_samples[:,column])
        Totalb = np.ones_like(samplesbulge_equal[:,column])
        Totalc = np.ones_like(samplesdisc_equal[:,column])
        Totala = Total_norm*Totala
        Totalb = Total_norm*Totalb
        Totalc =Total_norm*Totalc
        return countsT, edgesT, Total_norm, Totala, Totalb, Totalc
    
    @staticmethod
    def plotadjust(samplesbulge_equal, samplesdisc_equal, Total_samples, column,Totala, Totalb, Totalc, edgesT):
        plt.hist(samplesbulge_equal[:,column],weights=(samplesbulge_equal[:,4]/samplesbulge_equal[:,5])/Totalb,bins=edgesT, color='r', histtype='step', fill=False,label = 'Bulge')
        plt.hist(samplesdisc_equal[:,column],weights=(samplesdisc_equal[:,4]/samplesdisc_equal[:,5])/Totalc,bins=edgesT, color='b', histtype='step', fill=False,label = 'Disc')
        plt.hist(Total_samples[:,column],weights=(Total_samples[:,4]/Total_samples[:,5])/Totala,bins=edgesT, color='k', histtype='step', fill=False, label = 'Total')
        plt.show()
    
    @staticmethod
    def integralportion(samplesbulgeb_equal, samplesdiscb_equal, Total_samplesb, edgesT, Totala, Totalb, Totalc, column ):
        countsTx, edgesTx = np.histogram(Total_samplesb[:,column],weights=(Total_samplesb[:,4]/Total_samplesb[:,5])/Totala,bins=edgesT)
        countsBx, edgesBx = np.histogram(samplesbulgeb_equal[:,column],weights=(samplesbulgeb_equal[:,4]/samplesbulgeb_equal[:,5])/Totalb,bins=edgesT)
        countsDx, edgesDx = np.histogram(samplesdiscb_equal[:,column],weights=(samplesdiscb_equal[:,4]/samplesdiscb_equal[:,5])/Totalc,bins=edgesT)
        print('Total ',np.sum((countsTx) * np.diff(edgesTx)))
        print('Bulge ',np.sum((countsBx) * np.diff(edgesBx)))
        print('Disc ',np.sum((countsDx) * np.diff(edgesDx)))

if __name__ == '__main__':
    postprocessing = Postprocessing("./output/weighted_bulge_te", "./output/weighted_disk_te")
    postprocessing.full_postprocessing()