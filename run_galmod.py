"""
    Main module running the galactic model depending on the
    distribution.
    Options are 'bulge' or 'disc'
"""
import sys
from multiprocessing import Pool, cpu_count
import dynesty
from prior_definitions import prior_transform_bulge, prior_transform_disk
from likelihood_definitions import loglike
from galmod_constants import MEASURE_VAR
from utils import save_obj


def improved_run(prior_function, population_name, dlogz=0.1, ndim=6, npdim=4):
    """ simple main function for sampling. """
    # initialize our nested sampler
    print('Starting sampling')
    print('Measured variable mode ='+MEASURE_VAR)
    cpunum = cpu_count()
    with Pool(cpunum-1) as executor:
        sampler = dynesty.NestedSampler(loglike, prior_function,\
                ndim=ndim, npdim=npdim, pool=executor, queue_size=cpunum, bootstrap=0)
        sampler.run_nested(dlogz=dlogz, print_progress=True)#logl_max = -1e-10,dlogz = 1e-20)
        res = sampler.results
        res.summary()
        save_obj(res, './output/weighted_'+population_name+'_'+MEASURE_VAR)

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
