import sys

import galmoddefinitions as galfunc

import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import corner as cr
import dynesty
from dynesty import plotting as dyplot
import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

X_SOURCE = np.linspace(0, 1, 10000)
SIGMA_SOURCE_T = galfunc.source_bulge(X_SOURCE, 11000) #+ galfunc.source_disc(X_SOURCE, 11000)

#mass loads
bulgemasscdf = np.load('./bulgemasscdf.npy')
diskmasscdf = np.load('./diskmasscdf.npy')
xpdf = np.load('./massxpdf.npy')
#source
bulgemassdencdf = np.load('./bulgesourcedencdf.npy')
xpdfsource = np.load('./sourcedenxpdf.npy')

#lenses density
X_LENSES = np.linspace(0,1,1000)
def densitycdf(source_point):
    kbulge = galfunc.sigma_bulge(X_LENSES,source_point)
    kdisc = galfunc.sigma_disc(X_LENSES,source_point)
    k_bulge =  kbulge/(kbulge+kdisc)
    k_disc =  kdisc/(kbulge+kdisc)
    pdfbulge = galfunc.big_phi(X_LENSES, source_point, kbulge, 'bulge')
    pdfdisc = galfunc.big_phi(X_LENSES, source_point, kdisc, 'disk')
    pdf_total = (k_bulge*pdfbulge + k_disc*pdfdisc)
    cdfTotal = np.zeros_like(X_LENSES)
    cdfbulge = np.zeros_like(X_LENSES)
    cdfdisc = np.zeros_like(X_LENSES)
    for i,k in enumerate(X_LENSES):
        cdfTotal[i] = np.trapz(pdf_total[0:i], X_LENSES[0:i])
        cdfbulge[i] = np.trapz(pdfbulge[0:i], X_LENSES[0:i])
        cdfdisc[i] = np.trapz(pdfdisc[0:i], X_LENSES[0:i])
    
    return cdfTotal, cdfbulge, kbulge, cdfdisc, kdisc

def invdensitycdf(evalpoint,source_point):
    cdfTotal, cdfbulge, kbulge, cdfdisc, kdisc = densitycdf(source_point)
    k_total = kbulge + kdisc
    k_disc = kdisc/k_total
    invtotal = np.interp(evalpoint,cdfTotal,X_LENSES)
    invbulge = np.interp(evalpoint,cdfbulge,X_LENSES)
    invdisc = np.interp(evalpoint,cdfdisc,X_LENSES)
    return invtotal, invbulge, invdisc, kbulge, kdisc

vc= 100
#disk in km/s
VEL0_MEAN = 220.
NORMVEL0_DISP = VEL0_MEAN/vc
VEL_DISP_DISK = 30.
DISK_VAR_NORM = VEL_DISP_DISK/vc
#bulge in km/s
BULGE_MEAN_VEL = 0.
BULGE_MEAN_VEL_NORM = BULGE_MEAN_VEL/vc
VEL_DISP_BULGE = 100.
BULGE_VAR_NORM = VEL_DISP_BULGE/vc
xi_var = np.linspace(0,10,1000)
phi_vel_vector = np.vectorize(galfunc.phi_vel_s)

def cdfvel(x,source,population = 'bulge'):
    if population=='disk':
        normvelo0_disp = galfunc.v0abs(x, source, 359.56732242, 3.21595828)/vc
        vardisc = galfunc.varbulgedisc(x, VEL_DISP_BULGE, VEL_DISP_DISK)/vc
        velocity_vec = phi_vel_vector(xi_var, normvelo0_disp, vardisc)
    else:
        bulge_mean_vel_norm = 0.
        varbulge = galfunc.varbulgebulge(x, VEL_DISP_BULGE)/vc
        velocity_vec = phi_vel_vector(xi_var, bulge_mean_vel_norm, varbulge)
    cdfvel = np.zeros_like(xi_var)
    for i,k in enumerate(xi_var):
        cdfvel[i] = np.trapz(velocity_vec[0:i],xi_var[0:i])
    return cdfvel

def invcdfvel(evalpoint,x,source,population='bulge'):
    velcdf = cdfvel(x,source,population)
    invvel = np.interp(evalpoint, velcdf, xi_var)
    return invvel

def prior_transform_disk(unif_sphere):
    """Transforms the uniform random variable `unif_sphere ~ Unif[0., 1.)`
    to the parameter of interest `transformed_var`."""
    gal_prior = np.zeros((len(unif_sphere)+2))#np.array(unif_sphere).copy()
    gal_prior[0] = np.interp(unif_sphere[0], diskmasscdf, xpdf)  # Sample from the uniform sphere to the mass.
    gal_prior[1] = np.interp(unif_sphere[1], bulgemassdencdf, xpdfsource) # Sample from the uniform sphere to the bulge sources.
    _, _ , invdisc , kbulge , kdisc = invdensitycdf(unif_sphere[2],gal_prior[1])  # lenses density, it needs the  source value
    k_tot = kbulge + kdisc
    gal_prior[2] = invdisc 
    gal_prior[3] = invcdfvel(unif_sphere[3],gal_prior[2], gal_prior[1],'disk')  # velocity, it needs source and lens positions.
    gal_prior[4] = kdisc
    gal_prior[5] = k_tot
    return gal_prior

def loglike_disk_gal(sampled_val):
    mass = 10**sampled_val[0] # M/M_sun
    source_distance = sampled_val[1] #  sourdistance in parsecs
    x_ratios = sampled_val[2] # adimensional factor DL/source_distance
    norm_vel = sampled_val[3] # adimensional velocity v/v_c
    sigma_total = sampled_val[4] # This does not because of Gamma0
    re0c = galfunc.re0_ds(source_distance*3.086e16)
    gamma0c = 2*galfunc.OMEGA0*re0c*galfunc.vc*(1000)*sigma_total*((3.086e16)**(-2))
    omegac = gamma0c*norm_vel*np.sqrt(x_ratios*(1.-x_ratios))*(mass)**(-1./2.)*mass*np.log(10)
    proba = omegac
    if proba == 0.:
        return -np.inf
    else:
        return np.log(proba)

def improved_run_disk(dlogz=0.1, ndim=6,npdim=4):
    """ simple main function for sampling. """
    # initialize our nested sampler
    print('Starting sampling')
    cpunum = cpu_count() 
    with Pool(cpunum-1) as executor:
        sampler = dynesty.NestedSampler(loglike_disk_gal, prior_transform_disk, ndim=ndim, npdim=npdim,\
                pool=executor,
                queue_size = cpunum,
                bootstrap=0)
        sampler.run_nested(dlogz=dlogz, print_progress=True)#logl_max = -1e-10,dlogz = 1e-20)
        #sampler = dynesty.NestedSampler(loglike_bulge_gal,prior_transform_bulge, ndim)
        #sampler.run_nested(dlogz=dlogz)
        res = sampler.results
        res.summary()
        save_obj(res, 'sanity_check_disk3')
        #return res
        #fig, _ = dyplot.runplot(res, lnz_error=False)
        #fig1, _ = dyplot.traceplot(res, truths=np.zeros(ndim), \
        #                            truth_color='black', show_titles=True, \
        #                            trace_cmap='viridis', connect=True, \
        #                            connect_highlight=range(10))
        #fig2, _ = dyplot.cornerplot(res, color='blue', \
        #                        truth_color='black', show_titles=True, \
        #                        max_n_ticks=3, quantiles=None)
        #fig.savefig('./output/evidence.png')
        #fig1.savefig('./output/tracers.png')
        #fig2.savefig('./output/cornerplot.png')
        
if __name__ == '__main__': 
    improved_run_disk()
