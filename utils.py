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
    input:
        name: path with the name of the pickle without file extension.
    """
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Method to load pickle objects.
    input:
        name: path with the name of the pickle without file extension.
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
    """
        Class to postprocess and obtain the total and
        equally weighted samples produced by the main
        sampler.
        input:
            bulge_path: path and name of the file as required by
                    load_obj
            disk_path: path and name of the file as required by
                    load_obj
    """

    def __init__(self, bulge_path, disk_path):
        self.weightedouput_bulge = load_obj(bulge_path)
        self.weightedouput_disk = load_obj(disk_path)
        self.samplesbulge_equal = []
        self.samplesdisc_equal = []
        self.total_samples = []


    def full_postprocessing(self):
        """
        Method to run all the functions for default parameters for
        postprocesing the samples out of the bayesian analysis. It
        uses the 'output' folder as default path.
        """
        self.basic_postprocessing()
        np.save('./output/Totalsamples', self.total_samples)
        np.save('./output/bulge_samples_equal', self.samplesbulge_equal)
        np.save('./output/disk_samples_equal', self.samplesdisc_equal)
        self.plot_total_weighted(self.total_samples, "Total")
        self.plot_total_weighted(self.samplesbulge_equal, "Bulge")
        self.plot_total_weighted(self.samplesdisc_equal, "Disk")
        self.plot_individual_weighted(self.samplesbulge_equal, \
            self.samplesdisc_equal, self.total_samples)

    def basic_postprocessing(self):
        """
        Method to run the basic sampling uniform weigthing for the bayesian
        analysis. It also obtaines the total population sampling.
        It does not perform a population normalisation. The weights to produce
        the population ratio are included in indexes 4 and 5.
        """
        self.plot_unweighted(self.weightedouput_bulge, "bulge")
        self.plot_unweighted(self.weightedouput_disk, "disk")
        self.samplesbulge_equal, self.samplesdisc_equal = \
                                self.run_uniform_weight(self.weightedouput_bulge, \
                                                        self.weightedouput_disk)
        self.total_samples = np.vstack((self.samplesdisc_equal, self.samplesbulge_equal))

    @staticmethod
    def plot_unweighted(weightedouput, population):
        """
        Produces the corner, trace and runplot for the results out of the bayesian analysis
        using the individual populations without normalisation.
        input:
            weightedoutput: Result dictionary from the bayesian analysis.
            population: Label of the population to used.
        """
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
        """
        Corner plot of the galactic model using the uniformly weighted samples and
        normalizes to the total population.
        input:
            samples: numpy array containing the uniformly weighted samples.
            population: Label of the population to used.
        """
        figure = cr.corner(samples[:, :4], weights=samples[:, 4]/samples[:, 5], \
          labels=[r"$\log_{10}{M_L}$", r"$D_S$", r"$DL/DS$", r"$v/v_c$"], show_titles=True)
        figure.savefig("./output/Galmod_"+population+".png")

    @staticmethod
    def plot_individual_weighted(samplesbulge_equal, samplesdisc_equal, total_samples):
        """
        Histogram of the individual parameters comparing bulge,disk and total population.
        input:
            samplesbulge_equal: Numpy array of the bulge uniformely weighted samples.
            samplesdisc_equal: Numpy array of the disk uniformely weighted samples.
            total_samples: Numpy array of the combined samples forming the total population.
        """
        column = {"index":[0, 1, 2, 3], \
            "xlabel":[r"$\log_{10}{M_L/M_\odot}$", "kpc", " ", "(100 km/s)"], \
            "title":[r"$\log_{10}{M_L}$", r"$D_S$", r"$DL/DS$", r"$v/v_c$"], \
            "name":["Mass", "SourceDistance", "DistanceRatio", "Velocity"], \
            "scaling":[1, 1000, 1, 1]}
        for i in range(4):
            _, edgest, _, total_weight, total_bulge, total_disk = \
                Postprocessing.samplingadjust(samplesbulge_equal, samplesdisc_equal, \
                    total_samples, column["index"][i], column["scaling"][i])
            Postprocessing.plotadjust(samplesbulge_equal, samplesdisc_equal, \
            total_samples, total_weight, total_bulge, total_disk, edgest, column["index"][i], \
                column["xlabel"][i], column["title"][i], column["name"][i], \
                column["scaling"][i])

    @staticmethod
    def run_uniform_weight(unweightedouput_bulge, unweightedouput_disc):
        """
        Method to produce uniformly weighted samples from the bayesian analysis.
        input:
            unweightedouput_bulge: Dictionary of unweighted bulge samples.
            unweightedouput_disc: Dictionary of unweighted disk samples.
        output:
            samplesbulge_equal, samplesdisc_equal: Numpy array of uniformly weighted samples
            for bulge and disc respectively.
        """
        samplesbulge, weightsbulge = unweightedouput_bulge.samples, \
            np.exp(unweightedouput_bulge.logwt - unweightedouput_bulge.logz[-1])
        samplesbulge_equal = dyfunc.resample_equal(samplesbulge, weightsbulge)
        samplesdisc, weightsdisc = unweightedouput_disc.samples, \
            np.exp(unweightedouput_disc.logwt - unweightedouput_disc.logz[-1])
        samplesdisc_equal = dyfunc.resample_equal(samplesdisc, weightsdisc)
        return samplesbulge_equal, samplesdisc_equal

    @staticmethod
    def samplingadjust(samplesbulge_equal, samplesdisc_equal, total_samples, column, scaling=1):
        """
        Method for auxiliary weights to correctly plot histograms combining the different
        populations using the total population as  normalization.
        input:
            samplesbulge_equal: Numpy array of the bulge uniformely weighted samples.
            smaplesdisc_equal: Numpy array of the disk uniformely weighted samples.
            total_samples: Numpy array of the combined samples forming the total population.
            column: Index of the column in the Numpy array of the parameter to plot.
            scaling (optional): Scaling value for use of the correct units. Default is 1.
        output:
            countst: Number of counts on each bin of the histogram.
            edgest: Edges of each bin to use on the histogram.
            total_norm: Normalization value for the histogram bars.
            total_weight, total_bulge, total_disk: Normalization values for each bin
            with respect to the total population, bulge and disk respectively.
        """
        countst, edgest = np.histogram(total_samples[:, column]/scaling,\
             weights=total_samples[:, 4]/total_samples[:, 5], bins=30)
        total_norm = np.sum((countst) * np.diff(edgest))
        total_weight = np.ones_like(total_samples[:, column])
        total_bulge = np.ones_like(samplesbulge_equal[:, column])
        total_disk = np.ones_like(samplesdisc_equal[:, column])
        total_weight = total_norm*total_weight
        total_bulge = total_norm*total_bulge
        total_disk = total_norm*total_disk
        return countst, edgest, total_norm, total_weight, total_bulge, total_disk

    @staticmethod
    def plotadjust(samplesbulge_equal, samplesdisc_equal, \
        total_samples, total_weight, total_bulge, total_disk, \
        edgest, column, xlabel, title, name, sample_scaling=1):
        """
        Method to plot histograms combining the different
        populations using the total population as  normalization.
        input:
            samplesbulge_equal: Numpy array of the bulge uniformely weighted samples.
            samplesdisc_equal: Numpy array of the disk uniformely weighted samples.
            total_samples: Numpy array of the combined samples forming the total population.
            total_weight, total_bulge, total_disk: Normalization values for each bin
                with respect to the total population, bulge and disk respectively.
            edgest: Edges of each bin to use on the histogram.
            column: Index of the column in the Numpy array of the parameter to plot.
            xlabel: Label for the xaxis.
            title: Title of the plot.
            name: Name for the plot file.
            scaling (optional): Scaling value for use of the correct units. Default is 1.
        """
        fig, axis = plt.subplots()

        axis.hist(samplesbulge_equal[:, column]/sample_scaling, \
            weights=(samplesbulge_equal[:, 4]/samplesbulge_equal[:, 5])/total_bulge, bins=edgest, \
                color='r', histtype='step', fill=False, label='Bulge')
        axis.hist(samplesdisc_equal[:, column]/sample_scaling, \
            weights=(samplesdisc_equal[:, 4]/samplesdisc_equal[:, 5])/total_disk, bins=edgest, \
                color='b', histtype='step', fill=False, label='Disc')
        axis.hist(total_samples[:, column]/sample_scaling, \
            weights=(total_samples[:, 4]/total_samples[:, 5])/total_weight, bins=edgest, \
                color='k', histtype='step', fill=False, label='Total')
        axis.legend()
        axis.set_xlabel(xlabel)
        fig.suptitle(title)
        fig.savefig('./output/'+name+'_output.png')

    @staticmethod
    def integralportion(samplesbulgeb_equal, samplesdiscb_equal, \
        total_samplesb, edgest, total_weight, total_bulge, total_disk, column):
        """
        Method to calculate the integral of the distribution from the histogram normalization.
        Input:
            samplesbulgeb_equal: Numpy array of the bulge uniformely weighted samples.
            samplesdiscb_equal: Numpy array of the disk uniformely weighted samples.
            total_samplesb: Numpy array of the combined samples forming the total population.
            edgest: Edges of each bin to use on the histogram.
            total_weight, total_bulge, total_disk: Normalization values for each bin
                with respect to the total population, bulge and disk respectively.
            column: Index of the column in the Numpy array of the parameter to plot.
        """
        countstx, edgestx = np.histogram(total_samplesb[:, column], \
            weights=(total_samplesb[:, 4]/total_samplesb[:, 5])/total_weight, bins=edgest)
        countsbx, edgesbx = np.histogram(samplesbulgeb_equal[:, column], \
            weights=(samplesbulgeb_equal[:, 4]/samplesbulgeb_equal[:, 5])/total_bulge, bins=edgest)
        countsdx, edgesdx = np.histogram(samplesdiscb_equal[:, column], \
            weights=(samplesdiscb_equal[:, 4]/samplesdiscb_equal[:, 5])/total_disk, bins=edgest)
        print('Total ', np.sum((countstx) * np.diff(edgestx)))
        print('Bulge ', np.sum((countsbx) * np.diff(edgesbx)))
        print('Disc ', np.sum((countsdx) * np.diff(edgesdx)))

if __name__ == '__main__':
    postprocessing = Postprocessing("./output/weighted_bulge_te", "./output/weighted_disk_te")
    postprocessing.full_postprocessing()
