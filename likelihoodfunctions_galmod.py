""" Collection of probability likelihood gunctions for the galactic modeling """
import numpy as np
import galmoddefinitions as galfunc

X_SOURCE = np.linspace(0, 1, 10000)
SIGMA_SOURCE_T = galfunc.source_bulge(X_SOURCE, 11000) + galfunc.source_disc(X_SOURCE, 11000)

def te_ds(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma, sigma_total, \
    xval, val):
    """Returns the probability of a sampled value T_E by weighting from the
    T_E probability distribution of the data """
    if min(xval) < te_einstein <= max(xval):
        omegac = gamma*norm_vel*np.sqrt(x_ratios*(1.-x_ratios))*(mass)**(-1./2.)
        pte = np.interp(te_einstein, xval, val)
        #             print(pte)
        #             print(probulge_ds(omegac,source_distance,mass,norm_vel,\
        # x_ratios,sigma_total,var = "unfixed"))
        #             print(probdisk_ds(omegac,source_distance,mass,norm_vel,\
        # x_ratios,sigma_total,var = "unfixed"))
        #             print(Big_phi_source(source_distance,SIGMA_SOURCE_T))
        prob = galfunc.probulge_ds(omegac, source_distance, mass, norm_vel, x_ratios, \
                    sigma_total, var="unfixed")+\
                galfunc.probdisk_ds(omegac, source_distance, mass, norm_vel, x_ratios, \
                    sigma_total, var="unfixed")
        prob2 = galfunc.big_phi_source(source_distance, SIGMA_SOURCE_T)*pte
        #             print('interal' , prob)
        #             print('interal2' , prob2)
        return prob*prob2
    else:
        return 0.

def te_ds_bulge(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma, sigma_total, \
    xval, val):
    """Returns the probability of a sampled value T_E by weighting from the
    T_E probability distribution of the data. It only considers the bulge."""
    if min(xval) < te_einstein <= max(xval):
        omegac = gamma*norm_vel*np.sqrt(x_ratios*(1.-x_ratios))*(mass)**(-1./2.)
        pte = np.interp(te_einstein, xval, val)
#             print(pte)
#             print(probulge_ds(omegac,source_distance,mass,norm_vel,\
# x_ratios,sigma_total,var = "unfixed"))
#             print(probdisk_ds(omegac,source_distance,mass,norm_vel,\
# x_ratios,sigma_total,var = "unfixed"))
#             print(Big_phi_source(source_distance,SIGMA_SOURCE_T))
        prob = galfunc.probulge_ds(omegac, source_distance, mass, norm_vel, x_ratios, \
            sigma_total, var="unfixed")
        prob2 = galfunc.big_phi_source(source_distance, SIGMA_SOURCE_T)*pte
#             print('interal' , prob)
#             print('interal2' , prob2)
        return prob*prob2
    else:
        return 0.

def te_ds_disk(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma, sigma_total,\
    xval, val):
    """Returns the probability of a sampled value T_E by weighting from the
    T_E probability distribution of the data. It only considers the disk."""
    if min(xval) < te_einstein <= max(xval):
        omegac = gamma*norm_vel*np.sqrt(x_ratios*(1.-x_ratios))*(mass)**(-1./2.)
        pte = np.interp(te_einstein, xval, val)
#             print(pte)
#             print(probulge_ds(omegac,source_distance,mass,norm_vel,x_ratios,\
# sigma_total,var = "unfixed"))
#        print(probdisk_ds(omegac,source_distance,mass,norm_vel,x_ratios,\
# sigma_total,var = "unfixed"))
#         print(Big_phi_source(source_distance,SIGMA_SOURCE_T))
        prob = galfunc.probdisk_ds(omegac, source_distance, mass, norm_vel, x_ratios,\
            sigma_total, var="unfixed")
        prob2 = galfunc.big_phi_source(source_distance, SIGMA_SOURCE_T)*pte
#             print('interal' , prob)
#             print('interal2' , prob2)
        return prob*prob2
    else:
        return 0.

def te_ds_norm(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma, sigma_total, \
     xval, val):
    """Returns the probability of a sampled value T_E by weighting from the
    T_E probability distribution of the data. It trys to normalize z = 1."""
    if min(xval) < te_einstein <= max(xval):
        omegac = gamma*norm_vel*np.sqrt(x_ratios*(1.-x_ratios))*(mass)**(-1./2.)
        pte = np.interp(te_einstein, xval, val)
#             print(pte)
#             print(probulge_ds(omegac,source_distance,mass,norm_vel,x_ratios,\
# sigma_total,var = "unfixed"))
#             print(probdisk_ds(omegac,source_distance,mass,norm_vel,x_ratios,\
# sigma_total,var = "unfixed"))
#             print(Big_phi_source(source_distance,SIGMA_SOURCE_T))
        prob = (galfunc.probulge_ds(omegac, source_distance, mass, norm_vel, x_ratios, \
            sigma_total, var="unfixed")+ \
                galfunc.probdisk_ds(omegac, source_distance, mass, norm_vel, x_ratios, \
                    sigma_total, var="unfixed"))/np.exp(-39.552)
        prob2 = galfunc.big_phi_source(source_distance, SIGMA_SOURCE_T)*pte
#             print('interal' , prob)
#             print('interal2' , prob2)
        return prob*prob2
    else:
        return 0.

def pt_te(unif_sphere):
    """Transforms the uniform random variable `unif_sphere ~ Unif[0., 1.)`
    to the parameter of interest `transformed_var`."""
    transformed_var = np.array(unif_sphere).copy()
    transformed_var[0] = (1.8 -(-2))*unif_sphere[0] -2.  # scale and shift to [-2., 1.8)
    transformed_var[1] = 1.0*unif_sphere[1] # scale to [0., 1.)
    transformed_var[2] = 11000.*unif_sphere[2]  # scale and shift to [0., 10)
    transformed_var[3] = 10.*unif_sphere[3]  # normalized velocity [0, 10]
    return transformed_var

#This is the imputed probability. restricted by some of the delta functions.
def logprob_source2(sampled_val, xval, val):
    """Returns the log probability from the sampled values."""
    re0c = galfunc.re0_ds(sampled_val[2]*3.086e16)
    #eta = 5.*(vc*1000.)/((1./86400)*re0c)
    mass = 10**sampled_val[0] # M/M_sun
    x_ratios = sampled_val[1] # adimensional factor DL/source_distance
    source_distance = sampled_val[2] #  sourdistance in parsecs
    norm_vel = sampled_val[3] # adimensional velocity v/v_c
    sigma_total = 1. # This does not because of Gamma0
    gamma0c = 2*galfunc.OMEGA0*re0c*galfunc.vc*(1000)*sigma_total*((3.086e16)**(-2))
    eta = 2.*np.sqrt(mass)*np.sqrt(x_ratios*(1.-x_ratios))/norm_vel
    te_einstein = (1./86400)*re0c*eta/(galfunc.vc*1000.)
    prob = te_ds(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma0c, sigma_total,\
         xval, val)
    if prob == 0.:
        return -np.inf
    else:
        return np.log(prob)
def logprob_sourcebulge(sampled_val, xval, val):
    """Returns the log probability from the sampled values. Bulge only."""
    re0c = galfunc.re0_ds(sampled_val[2]*3.086e16)
    #eta = 5.*(vc*1000.)/((1./86400)*re0c)
    mass = 10**sampled_val[0] # M/M_sun
    x_ratios = sampled_val[1] # adimensional factor DL/source_distance
    source_distance = sampled_val[2] #  sourdistance in parsecs
    norm_vel = sampled_val[3] # adimensional velocity v/v_c
    sigma_total = 1. # This does not because of Gamma0
    gamma0c = 2*galfunc.OMEGA0*re0c*galfunc.vc*(1000)*sigma_total*((3.086e16)**(-2))
    eta = 2.*np.sqrt(mass)*np.sqrt(x_ratios*(1.-x_ratios))/norm_vel
    te_einstein = (1./86400)*re0c*eta/(galfunc.vc*1000.)
    prob = te_ds_bulge(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma0c, \
        sigma_total, xval, val)
    if prob == 0.:
        return -np.inf
    else:
        return np.log(prob)

def logprob_sourcedisk(sampled_val, xval, val):
    """Returns the log probability from the sampled values. Disk only."""
    re0c = galfunc.re0_ds(sampled_val[2]*3.086e16)
    #eta = 5.*(vc*1000.)/((1./86400)*re0c)
    mass = 10**sampled_val[0] # M/M_sun
    x_ratios = sampled_val[1] # adimensional factor DL/source_distance
    source_distance = sampled_val[2] #  sourdistance in parsecs
    norm_vel = sampled_val[3] # adimensional velocity v/v_c
    sigma_total = 1. # This does not because of Gamma0
    gamma0c = 2*galfunc.OMEGA0*re0c*galfunc.vc*(1000)*sigma_total*((3.086e16)**(-2))
    eta = 2.*np.sqrt(mass)*np.sqrt(x_ratios*(1.-x_ratios))/norm_vel
    te_einstein = (1./86400)*re0c*eta/(galfunc.vc*1000.)
    prob = te_ds_disk(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma0c, \
        sigma_total, xval, val)
    if prob == 0.:
        return -np.inf
    else:
        return np.log(prob)

def logprob_sourcenorm(sampled_val, xval, val):
    """Returns the log probability from the sampled values. Tries to use normalised z."""
    re0c = galfunc.re0_ds(sampled_val[2]*3.086e16)
    #eta = 5.*(vc*1000.)/((1./86400)*re0c)
    mass = 10**sampled_val[0] # M/M_sun
    x_ratios = sampled_val[1] # adimensional factor DL/source_distance
    source_distance = sampled_val[2] #  sourdistance in parsecs
    norm_vel = sampled_val[3] # adimensional velocity v/v_c
    sigma_total = 1. # This does not because of Gamma0
    gamma0c = 2*galfunc.OMEGA0*re0c*galfunc.vc*(1000)*sigma_total*((3.086e16)**(-2))
    eta = 2.*np.sqrt(mass)*np.sqrt(x_ratios*(1.-x_ratios))/norm_vel
    te_einstein = (1./86400)*re0c*eta/(galfunc.vc*1000.)
    prob = te_ds_norm(mass, norm_vel, x_ratios, source_distance, te_einstein, gamma0c, sigma_total,\
        xval, val)
    if prob == 0.:
        return -np.inf
    else:
        return np.log(prob)
