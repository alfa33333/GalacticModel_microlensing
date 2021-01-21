""" Likelihood functions and definitions for the Bayesian Analysis
    of the Galactic Model.
"""
import numpy as np
import galmoddefinitions as galfunc
from galmod_constants import THETA_STAR, TE_MEAN, TE_ERROR, RHO_MEAN, RHO_ERROR, MEASURE_VAR

def diff_chi2(rho):
    """ Gaussian distribution given by the difference chi2 from the measured 'rho'.
        input:
            rho: expected 'te' .
        return:
            weight: probability given a the 'rho' value.
    """
    prob = (1/(RHO_ERROR*np.sqrt(2*np.pi)))*np.exp(-(rho-RHO_MEAN)**2/(2*RHO_ERROR**2))
    return prob

def calculated_weight(te_einstein):
    """ Gaussian distribution for with the measured 'te'.
        input:
            te_einstein: expected 'te' .
        return:
            weight: probability given a the te value.
    """
    weight = (1/(TE_ERROR*np.sqrt(2*np.pi)))*np.exp(-(te_einstein-TE_MEAN)**2/(2*TE_ERROR**2))
    return weight

def loglike(sampled_val):
    """ Likelihood function for Bayesian analysis
    input:
        sampled_val: Transformed values from uniform sphere.
    return:
        logl:  loglikelihood for the parameters.
    """
    mass = 10**sampled_val[0] # M/M_sun
    source_distance = sampled_val[1] #  sourdistance in parsecs
    x_ratios = sampled_val[2] # adimensional factor DL/source_distance
    norm_vel = sampled_val[3] # adimensional velocity v/v_c
    sigma_total = sampled_val[4] # This does not because of Gamma0
    re0c = galfunc.re0_ds(source_distance*3.086e16)
    gamma0c = 2*galfunc.OMEGA0*re0c*galfunc.vc*(1000)*sigma_total*((3.086e16)**(-2))
    omegac = gamma0c*norm_vel*np.sqrt(x_ratios*(1.-x_ratios))*(mass)**(-1./2.)*mass*np.log(10)
    eta = 2.*np.sqrt(mass)*np.sqrt(x_ratios*(1.-x_ratios))/norm_vel
    te_einstein = (1./86400)*re0c*eta/(galfunc.vc*1000.)
    weight = calculated_weight(te_einstein)
    if MEASURE_VAR == 'TE':
        proba = weight*omegac
    elif MEASURE_VAR == 'TE_RHO':
        thetae = galfunc.thetae_func(np.array([np.log10(mass), x_ratios, source_distance]))
        if thetae == 0.0:
            rho = 10000000
        else:
            rho = THETA_STAR/thetae
        prho = diff_chi2(rho)
        proba = weight*omegac*prho
    if proba == 0.:
        return -np.inf
    else:
        logl = np.log(proba)
        return logl
