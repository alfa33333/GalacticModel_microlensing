"""
    Main module running the galactic model depending on the
    distribution.
    Options are 'bulge' or 'disc'
"""
import sys
from multiprocessing import Pool, cpu_count
import dynesty
import numpy as np
from prior_definitions import prior_transform_bulge, prior_transform_disk, save_obj
import galmoddefinitions as galfunc

tE_best = 6.274
tE_error = 0.057

rho_best = 0.060464435134475955
rho_error = 0.0006404327321859689

def DCHI2(rho):
    prob = (1/(rho_error*np.sqrt(2*np.pi)))*np.exp(-(rho-rho_best)**2/(2*rho_error**2))
    return prob

def weightte(tE):
    return (1/(tE_error*np.sqrt(2*np.pi)))*np.exp(-(tE-tE_best)**2/(2*tE_error**2))

def loglike(sampled_val):
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
    weight = (1/(tE_error*np.sqrt(2*np.pi)))*np.exp(-(te_einstein-tE_best)**2/(2*tE_error**2))
#     thetae = galfunc.thetae_func(np.array([np.log10(mass), x_ratios, source_distance]))
#     theta_star = 16.2
#     if thetae == 0.0:
#         rho = 10000000
#     else:
#         rho = theta_star/thetae
#     prho = DCHI2(rho)
    proba = weight*omegac#*prho
    if proba == 0.:
        return -np.inf
    else:
        logl = np.log(proba)
        return logl

def improved_run(prior_function, population_name, dlogz=0.1, ndim=6, npdim=4):
    """ simple main function for sampling. """
    # initialize our nested sampler
    print('Starting sampling')
    cpunum = cpu_count()
    with Pool(cpunum-1) as executor:
        sampler = dynesty.NestedSampler(loglike, prior_function,\
                ndim=ndim, npdim=npdim, pool=executor, queue_size=cpunum, bootstrap=0)
        sampler.run_nested(dlogz=dlogz, print_progress=True)#logl_max = -1e-10,dlogz = 1e-20)
        res = sampler.results
        res.summary()
        save_obj(res, './output/weighted_'+population_name+'_te')

def main(parameters):
    """
    Method to select the population for the nested sampling.
    """
    population = parameters['population']
    dlogz = parameters['dlogz']  if 'dlogz' in parameters.keys() else 0.1
    ndim = parameters['ndim']  if 'ndim' in parameters.keys() else 6
    npdim = parameters['npdim']  if 'npdim' in parameters.keys() else 4
    print('Population lens: {}'.format(population))
    print('Sampler parameters: dlogz = {}, ndim = {}, npdim = {}'.format(dlogz, ndim, npdim))
    if population == 'disc':
        improved_run(prior_function=prior_transform_disk, population_name='disc', \
            dlogz=0.1, ndim=6, npdim=4)
    elif population == 'bulge':
        improved_run(prior_function=prior_transform_bulge, population_name='bulge',\
            dlogz=0.1, ndim=6, npdim=4)
        # improved_run_bulge(dlogz=dlogz, ndim=ndim, npdim=npdim)

if __name__ == '__main__':
    ARGUMENT = sys.argv
    DYNESTY_OPT = ['population', 'dlogz', 'ndim', 'npdim']
    op = dict(zip(DYNESTY_OPT, ARGUMENT[1:]))
    if op['population'] in ('disc', 'bulge'):
        main(op)
    else:
        print("No type of distribution specified.")
