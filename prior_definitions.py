import pickle
import numpy as np
import galmoddefinitions as galfunc


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as file:
        return pickle.load(file)

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
X_LENSES = np.linspace(0, 1, 1000)
def densitycdf(source_point):
    kbulge = galfunc.sigma_bulge(X_LENSES, source_point)
    kdisc = galfunc.sigma_disc(X_LENSES, source_point)
    k_bulge = kbulge/(kbulge+kdisc)
    k_disc = kdisc/(kbulge+kdisc)
    pdfbulge = galfunc.big_phi(X_LENSES, source_point, kbulge, 'bulge')
    pdfdisc = galfunc.big_phi(X_LENSES, source_point, kdisc, 'disk')
    pdf_total = (k_bulge*pdfbulge + k_disc*pdfdisc)
    # pdf_Bulge = k_bulge*pdfbulge
    # pdf_disc  = k_disc*pdfdisc
    cdftotal = np.zeros_like(X_LENSES)
    cdfbulge = np.zeros_like(X_LENSES)
    cdfdisc = np.zeros_like(X_LENSES)
    for i in enumerate(X_LENSES):
        cdftotal[i] = np.trapz(pdf_total[0:i], X_LENSES[0:i])
        cdfbulge[i] = np.trapz(pdfbulge[0:i], X_LENSES[0:i])
        cdfdisc[i] = np.trapz(pdfdisc[0:i], X_LENSES[0:i])
    return cdftotal, cdfbulge, kbulge, cdfdisc, kdisc

def invdensitycdf(evalpoint, source_point):
    cdftotal, cdfbulge, kbulge, cdfdisc, kdisc = densitycdf(source_point)
    invtotal = np.interp(evalpoint, cdftotal, X_LENSES)
    invbulge = np.interp(evalpoint, cdfbulge, X_LENSES)
    invdisc = np.interp(evalpoint, cdfdisc, X_LENSES)
    return invtotal, invbulge, invdisc, kbulge, kdisc

vc = 100
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
xi_var = np.linspace(0, 10, 1000)
phi_vel_vector = np.vectorize(galfunc.phi_vel_s)

def cdfvel(x_ratio, source, population='bulge'):
    """ Numerical CDF for the velocity probability
    input:
       x_ratio: Source-Lens distance ratio DL/DS
        source: source density
        population: Population of the lens
            'bulge' or 'disk'
    return:
        cdfvel_vec: vector containing the CDF values.
    """
    if population == 'disk':
        normvelo0_disp = galfunc.v0abs(x_ratio, source, 359.56732242, 3.21595828)/vc
        vardisc = galfunc.varbulgedisc(x_ratio, VEL_DISP_BULGE, VEL_DISP_DISK)/vc
        velocity_vec = phi_vel_vector(xi_var, normvelo0_disp, vardisc)
    else:
        bulge_mean_vel_norm = 0.
        varbulge = galfunc.varbulgebulge(x_ratio, VEL_DISP_BULGE)/vc
        velocity_vec = phi_vel_vector(xi_var, bulge_mean_vel_norm, varbulge)
    cdfvel_vec = np.zeros_like(xi_var)
    for i in enumerate(xi_var):
        cdfvel_vec[i] = np.trapz(velocity_vec[0:i], xi_var[0:i])
    return cdfvel_vec

def invcdfvel(evalpoint, x_ratio, source, population='bulge'):
    """ Numerically inverse the velocity CDF.
    input:
        evalpoint: Point for evaluation
        x_ratio: Source-Lens distance ratio DL/DS
        source: source density
        population: Population of the lens
            'bulge' or 'disk'
    return:
        invvel: inverse velocity CDF
    """
    velcdf = cdfvel(x_ratio, source, population)
    invvel = np.interp(evalpoint, velcdf, xi_var)
    return invvel

def prior_transform_bulge(unif_sphere):
    """Transforms the uniform random variable `unif_sphere ~ Unif[0., 1.)`
    to the parameter of interest `transformed_var` for bulge lenses."""
    gal_prior = np.zeros((len(unif_sphere)+2))
    # Sample from the uniform sphere to the mass.
    gal_prior[0] = np.interp(unif_sphere[0], bulgemasscdf, xpdf)
    # Sample from the uniform sphere to the bulge sources.
    gal_prior[1] = np.interp(unif_sphere[1], bulgemassdencdf, xpdfsource)
    # lenses density, it needs the  source value.
    _, invbulge, _, kbulge, kdisc = invdensitycdf(unif_sphere[2], gal_prior[1])
    k_tot = kbulge + kdisc
    gal_prior[2] = invbulge
    # velocity, it needs source and lens positions.
    gal_prior[3] = invcdfvel(unif_sphere[3], gal_prior[2], gal_prior[1], 'bulge')
    gal_prior[4] = kbulge
    gal_prior[5] = k_tot
    return gal_prior

def prior_transform_disk(unif_sphere):
    """Transforms the uniform random variable `unif_sphere ~ Unif[0., 1.)`
    to the parameter of interest `transformed_var` for the disc lenses."""
    gal_prior = np.zeros((len(unif_sphere)+2))
    # Sample from the uniform sphere to the mass.
    gal_prior[0] = np.interp(unif_sphere[0], diskmasscdf, xpdf)
    # Sample from the uniform sphere to the bulge sources.
    gal_prior[1] = np.interp(unif_sphere[1], bulgemassdencdf, xpdfsource)
    # lenses density, it needs the  source value
    _, _, invdisc, kbulge, kdisc = invdensitycdf(unif_sphere[2], gal_prior[1])
    k_tot = kbulge + kdisc
    gal_prior[2] = invdisc
    # velocity, it needs source and lens positions.
    gal_prior[3] = invcdfvel(unif_sphere[3], gal_prior[2], gal_prior[1], 'disk')
    gal_prior[4] = kdisc
    gal_prior[5] = k_tot
    return gal_prior
