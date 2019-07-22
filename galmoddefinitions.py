''' This module contains all the Galactic model function definitions
    and some useful constants as:
    NEWTON_G = Newtons constant
    c = speed of light
    M_sun = solar mass
    GM_sun = Newtons constant * solar mass
    OMEGA0 = internal omega_0 constant
    k_G = sqrt( G * M / c ^2)
    vc = velocity normalization constant in km/s

    All the functions defined here can be found described in:
    M. Dominik, 2005.	arXiv:astro-ph/0507540

    '''
import numpy as np
from astropy import constants as const
from astropy import units as u

#constants + vc
NEWTON_G = const.G.value
c = const.c.value # pylint: disable=invalid-name
M_sun = const.M_sun.value # pylint: disable=invalid-name
GM_sun = const.GM_sun.value # pylint: disable=invalid-name
OMEGA0 = 2.0
k_G = np.sqrt(GM_sun/c**2) # pylint: disable=invalid-name
vc = 100  # pylint: disable=invalid-name


def density_bulge(d_lens):
    """ Calculates the density for the Bulge for a particular lens distance.
        The galactic center is hardcoded to 8500pc.
        input:
            d_lens :  distance of the lens in parsecs
        output:
            density_bulge_out : density of the bulge.
    """

    r0_center = 8500
    lr_angle = np.radians(358.38852549)
    br_angle = np.radians(-3.31374594)
    xl_distance = d_lens*np.cos(lr_angle)*np.cos(br_angle)-r0_center
    yl_distance = d_lens*np.sin(lr_angle)*np.cos(br_angle)
    zl_distance = d_lens*np.sin(br_angle)
    rot = np.radians(20)
    xldash = xl_distance*np.cos(rot) + np.sin(rot)*yl_distance
    yldash = -xl_distance*np.sin(rot) + yl_distance*np.cos(rot)
    zldash = zl_distance
    rho_0 = 2.1 #M_sun pc-3
    norm_dist = np.sqrt(((xldash/1580)**2+(yldash/620)**2)**2+(zldash/430)**4)
    density_bulge_out = (rho_0*np.exp(-0.5*norm_dist))
    return density_bulge_out

def density_disc_lens(d_lens):
    """ Calculates the density for a lens in the disc:
        The galactic center distance is hardcoded to 8500.
        input:
            d_lens : distance to the lens.
        output:
            density : the density of lenses at input distance.
    """
    r0_center = 8500
    lr_angle = np.radians(358.38852549)
    br_angle = np.radians(-3.31374594)
    zl_distance = d_lens*np.sin(br_angle)
    radius_cyl_dist = r0_center*np.sqrt((np.cos(lr_angle)-d_lens*np.cos(br_angle)/r0_center)**2+\
        np.sin(lr_angle)**2)
    thick_disc = (35./1000.) * np.exp(-np.abs(zl_distance)/1000.) # units of M_sun *pc^-3
    thin_disc = (25./300.) * np.exp(-np.abs(zl_distance)/300.)
    density = 0.5*np.exp(-(radius_cyl_dist-r0_center)/3500)*(thick_disc+thin_disc)
    return density

DENSITY_BULGE_VECTOR = np.vectorize(density_bulge)

def sigma_bulge(x_ratios, distance_source):
    """ Returns the surface density bulge component"""
    return distance_source*np.trapz(DENSITY_BULGE_VECTOR(x_ratios*distance_source), x_ratios)

def sigma_disc(x_ratios, distance_source):
    """ Returns the surface density disk component"""
    return distance_source*np.trapz(density_disc_lens(x_ratios*distance_source), x_ratios)

def big_phi(x_ratios, distance_source, big_sigma, site):
    """ Returns the normalised density function:
        Options for site:
        'bulge' : only bulge part
        'disk'  : only disc part
        'total' : bulge + disc
    """
    if site == 'bulge':
        return distance_source*DENSITY_BULGE_VECTOR(x_ratios*distance_source)/big_sigma
    elif site == 'disk':
        return distance_source*density_disc_lens(x_ratios*distance_source)/big_sigma
    elif site == 'total':
        return distance_source*(DENSITY_BULGE_VECTOR(x_ratios*distance_source)+\
            density_disc_lens(x_ratios*distance_source))/big_sigma
    else:
        #This will assume bulge as default.
        return distance_source*DENSITY_BULGE_VECTOR(x_ratios*distance_source)/big_sigma
#Extradded
######################
def big_radius(d_lens, lr_angle, br_angle):
    """
    Calculates the cylindrical distance from the galactic center as :
    R = sqr( (X-R_0)^2 + Y^2)   )
    """
    r0_center = 8500
    lr_angle = np.radians(lr_angle)
    br_angle = np.radians(br_angle)
    radius_cyl_dist = r0_center*np.sqrt((np.cos(lr_angle)-d_lens*np.cos(br_angle)/r0_center)**2+\
        np.sin(lr_angle)**2)
    return radius_cyl_dist

def vcirc(cyl_dist_r):
    """ Returs circular velocity from cylindrical distance """
    return 220*4.23*np.sqrt(8.5e3/cyl_dist_r)*np.sqrt(np.log(1.+cyl_dist_r/20e3)-\
        cyl_dist_r/(cyl_dist_r+20e3)) #km/s

def voy(lr_angle):
    """ Returns y component for reference velocity v_0"""
    return -np.sin(np.radians(lr_angle))*9+np.cos(np.radians(lr_angle))*(12.+220.)
def voz(lr_angle, br_angle):
    """ Returns z component for reference velocity v_0"""
    return -np.cos(np.radians(lr_angle))*np.sin(np.radians(br_angle))*9-\
        np.sin(np.radians(lr_angle))*np.sin(np.sin(np.radians(br_angle)))*(12.+220.) +\
             np.cos(np.radians(br_angle))*7.

def vly(x_ratio, distance_source, lr_angle, br_angle):
    """ Returns y component for lens velocity of reference v_0"""
    lens_dist = x_ratio*distance_source
    return vcirc(big_radius(x_ratio, lr_angle, br_angle))*(8.5e3*np.cos(np.radians(lr_angle))-\
        lens_dist*np.cos(np.radians(br_angle)))/big_radius(x_ratio, lr_angle, br_angle)

def vlz(x_ratio, lr_angle, br_angle):
    """ Returns z component for lens velocity of reference v_0"""
    vel_circ = vcirc(big_radius(x_ratio, lr_angle, br_angle))
    cyl_distance_radius = big_radius(x_ratio, lr_angle, br_angle)
    radi_project = 8.5e3*np.sin(np.radians(lr_angle))*np.sin(np.radians(br_angle))
    return vel_circ*radi_project/cyl_distance_radius

def v0abs(x_ratio, distance_source, lr_angle, br_angle):
    """ Returns the norm for the velocity reference v_0"""
    vy_comp = vly(x_ratio, distance_source, lr_angle, br_angle) - (1.-x_ratio)*voy(lr_angle)
    vz_comp = vlz(x_ratio, lr_angle, br_angle) - (1.-x_ratio)*voz(lr_angle, br_angle)
    #vy = - (1.-x)*voy(lr_angle,b)
    #vz = - (1.-x)*voz(lr_angle,b)
    return np.sqrt(vy_comp**2+vz_comp**2)


##############################
def phi_vel_s(perp_vel, perp_vel_ref, var):
    """ Returns the probability density of the absolute effective velocity:
        input:
            perp_vel :  perpendicula velocity
            perp_vel_ref: reference perpendicular velocity or v_0
            var : variance or sigma
        output:
            probability density"""
    arg_bessel = perp_vel*perp_vel_ref/var**2
#     print(arg_bessel)
#     print((perp_vel/var**2))
#     print(np.exp(-(perp_vel**2+perp_vel_ref**2)/(2.*var**2)))
    if arg_bessel > 700.:
        arg_bessel = 700.
    else:
        arg_bessel = perp_vel*perp_vel_ref/var**2
    npio = np.i0(arg_bessel)
    #if npio == np.inf:
    #    return np.inf
    #else:
    return (perp_vel/var**2)*np.exp(-(perp_vel**2+perp_vel_ref**2)/(2.*var**2))*npio

def varbulgebulge(x_ratio, bulge):
    """ Returns the total velocity dispersion of a bulge lens with a bulge source. """
    return np.sqrt(1.+x_ratio**2)*bulge

def varbulgedisc(x_ratio, bulge, disc):
    """ Returns the totals velocity dispersion of a disk lens with a bulge source """
    return np.sqrt(bulge**2*x_ratio**2 + disc**2)

#disk in km/s
VEL0_MEAN = 220.
NORMVEL0_DISP = VEL0_MEAN/vc
VEL_DISP_DISK = 30.
DISK_VAR_NORM = VEL_DISP_DISK/vc
#bulge in km/s
BULGE_MEAN_VEL = 0.
BULGE_MEAN_VEL_NORM = BULGE_MEAN_VEL/vc
VEL_DISP_DISK = 100.
BULGE_VAR_NORM = VEL_DISP_DISK/vc

def genpowlaw(base, arg, power):
    """Generic power law with inverse power:
        arg*base ^ (- power)
    """
    return arg*base**(-power)

def genexplaw(log10_value, arg, char_log10val, sigma):
    """returns the result of a Generic exponential law for Chabrier"""
    return arg*(np.exp(-0.5*(np.log10(log10_value) - np.log10(char_log10val))**2/(sigma**2)))

def logmass_spectrum_bulge(log10_mass):
    '''logMass function for the bulge'''
    intconst = 0.94129 #Constant of integration to 1.
    if -2 <= np.log10(log10_mass) < -0.155:
        return intconst*genexplaw(log10_mass, 1/(0.33*np.sqrt(2*np.pi)), 0.22, 0.33)
    elif -0.155 <= np.log10(log10_mass) <= 1.8:
        matchconst = 0.238365
        return intconst*matchconst*genpowlaw(log10_mass, 1, 1.3)
    else:
        return 0.0

def logmass_spectrum_disc(log10_mass):
    '''Mass function for the disk'''
    intconst = 0.937672883437883 #Constant of integration to 1.
    if -2 <= np.log10(log10_mass) < -0.7:
        return intconst*genpowlaw(log10_mass, 1, -0.2)
    elif -0.7 <= np.log10(log10_mass) < 0.0:
        matchconst = 1.48472 #constant for proper match
        argument = 1/(0.69*np.sqrt(2*np.pi))
        return intconst*matchconst*genexplaw(log10_mass, argument, 10**-1.102, 0.69)
    elif (np.log10(log10_mass) >= 0.0) and (np.log10(log10_mass) < 0.54):
        matchconst = 0.239785
        return intconst*matchconst*genpowlaw(log10_mass, 1, 4.37)
    elif (np.log10(log10_mass >= 0.54)) and (np.log10(log10_mass) < 1.26):
        matchconst = 0.0843766
        return intconst*matchconst*genpowlaw(log10_mass, 1, 3.53)
    elif (np.log10(log10_mass) >= 1.26) and (np.log10(log10_mass) <= 1.8):
        matchconst = 0.00137095
        return intconst*matchconst*genpowlaw(log10_mass, 1, 2.11)
    else:
        return 0.

def big_phi_source(distance_source, surface_density):
    """ Returns the normalised probability density for the source positions """
    return ((distance_source*DENSITY_BULGE_VECTOR(distance_source))+\
        (distance_source*density_disc_lens(distance_source)))/surface_density

def source_bulge(x_ratios, distance_source):
    """ Returns the surface density for the bulge component"""
    distance_vec = x_ratios*distance_source
    return np.trapz(distance_vec*DENSITY_BULGE_VECTOR(distance_vec), distance_vec)

def source_disc(x_ratios, distance_source):
    """ Returns the surface density for the disk component"""
    distance_vec = x_ratios*distance_source
    return np.trapz((distance_vec)*density_disc_lens(distance_vec), distance_vec)


#Full selection
def re0_ds(distance_source):
    """Returns the Einstein Radius of 1 solar mass located half-way
    between observer and source"""
    return k_G*np.sqrt(distance_source)

def sigma_total_lens(distance_source, x_lin_dist):
    """Estimates the total surface mass density. This includes bulk and disk"""
    return sigma_bulge(x_lin_dist, distance_source) + sigma_disc(x_lin_dist, distance_source)

def gamma0_ds(distance_source, x_lin_dist):
    """Normalisation constant for the event rate Gamma"""
    omega0 = 2.
    return 2*omega0*re0_ds(distance_source)*vc*sigma_total_lens(distance_source, x_lin_dist)

def omega_s(distance_source, mass, xi_var, x_ratios, x_lin_dist):
    """Returns the evaluated kernel OMEGA"""
    gamma0_const = gamma0_ds(distance_source, x_lin_dist)
    return gamma0_const*xi_var*np.sqrt(x_ratios*(1.-x_ratios))*(mass)**(-1./2.)

def probulge_ds(omega, distance_source, mass, xi_var, x_ratios, sigma_total, var='fixed'):
    """ Returns the probability of the the lens in the bulge at a given source distance

        the variance by variable var can be chosen to be 'fixed' or 'unfixed'
    """
    #print(big_phi(x_ratios,distance_source,sigma_total,'bulge'))
    #print(phi_vel_s(xi_var,BULGE_MEAN_VEL_NORM,BULGE_VAR_NORM))
    #print(logmass_spectrum_bulge(mass)/(mass*np.log(10)))
    bulge_mean_vel_norm = 0.
    if var == 'fixed':
        #return omega*big_phi(x_ratios,distance_source,sigma_total,'bulge')*\
        # phi_vel_s(xi_var,bulge_mean_vel_norm,BULGE_VAR_NORM)*\
        # logmass_spectrum_bulge(mass)/(mass*np.log(10))
        spectrum = logmass_spectrum_bulge(mass)
        velocity_norm = phi_vel_s(xi_var, bulge_mean_vel_norm, BULGE_VAR_NORM)
        density_probability = big_phi(x_ratios, distance_source, sigma_total, 'bulge')
        return omega*density_probability*velocity_norm*spectrum
    else:
        varbulge = varbulgebulge(x_ratios, VEL_DISP_DISK)/vc
#         print('var',varbulge)
#         print('big',big_phi(x_ratios,distance_source,sigma_total,'bulge'))
#         print('vel',np.log(phi_vel_s(xi_var,bulge_mean_vel_norm,varbulge)))
#         print('log',logmass_spectrum_bulge(mass)/(mass*np.log(10)))
        #return omega*big_phi(x_ratios,distance_source,sigma_total,'bulge')*\
        # phi_vel_s(xi_var,bulge_mean_vel_norm,varbulge)*\
        # logmass_spectrum_bulge(mass)/(mass*np.log(10))
        spectrum = logmass_spectrum_bulge(mass)
        velocity_norm = phi_vel_s(xi_var, bulge_mean_vel_norm, varbulge)
        density_probability = big_phi(x_ratios, distance_source, sigma_total, 'bulge')
        return omega*density_probability*velocity_norm*spectrum

def probdisk_ds(omega, distance_source, mass, xi_var, x_ratios, sigma_total, var='fixed'):
    """ Returns the probability of the the lens in the disk at a given source distance

        the variance by variable var can be chosen to be 'fixed' or 'unfixed'
    """
    normvelo0_disp = v0abs(x_ratios, distance_source, 358.38852549, -3.31374594)
    #print(big_phi(x_ratios,distance_source,sigma_total,'disk'))
    #print(phi_vel_s(xi_var,normvelo0_disp,DISK_VAR_NORM))
    #print(logmass_spectrum_disc(mass)/(mass*np.log(10)))

    if var == 'fixed':
        #return omega*big_phi(x_ratios,distance_source,sigma_total,'disk')*\
        # phi_vel_s(xi_var,normvelo0_disp,DISK_VAR_NORM)*\
        # logmass_spectrum_disc(mass)/(mass*np.log(10))
        spectrum = logmass_spectrum_disc(mass)
        velocity_norm = phi_vel_s(xi_var, normvelo0_disp, DISK_VAR_NORM)
        density_probability = big_phi(x_ratios, distance_source, sigma_total, 'disk')
        return omega*density_probability*velocity_norm*spectrum
    else:
        vardisc = varbulgedisc(x_ratios, VEL_DISP_DISK, VEL_DISP_DISK)/vc
        #print(vardisc)
        #return omega*big_phi(x_ratios,distance_source,sigma_total,'disk')*\
        # phi_vel_s(xi_var,normvelo0_disp,vardisc)*\
        # logmass_spectrum_disc(mass)/(mass*np.log(10))
        spectrum = logmass_spectrum_disc(mass)
        velocity_norm = phi_vel_s(xi_var, normvelo0_disp, vardisc)
        density_probability = big_phi(x_ratios, distance_source, sigma_total, 'disk')
        return omega*density_probability*velocity_norm*spectrum

#samples reconstruction
def cumdev(data, x_points):
    """Returns the accumulative data from the derivative. """
    acum = np.zeros(len(data))
    for i in range(len(data)-1):
        acum[i+1] = (data[i+1]-data[i])/(x_points[i+1]-x_points[i])
    return acum

def probte(file, column, nbins=32):
    """Returns the simple reconstruction of the probability from the data histogram.
    In numpy array format.
    input:
        file : File to load
        column : The column from where the probability is going to be reconstructed
        nbins (optional) : The number of bins to be used. Defaul = 32.

    returns:

    """
    samples = np.load(file)
    tesamples = samples[:, column]
    values, base = np.histogram(tesamples, bins=nbins)
    cumulative = np.cumsum(values)
    prob_val = cumdev(cumulative/len(tesamples), base[:-1])
    x_axis = base[:-1]
    return x_axis, prob_val

### derived definitions
#############################
def thetae_func(samples):
    """ Returns the Einstein angle in microarcseconds.
    Input:
        samples : Input numpy array containing Mass in solar mass,
        the ratios of the lens distance and source distance in parsecs
        and  the distance to the source in parsecs.
        This could also be a 2d array where each column is  mass, ratios
        and source distance.

    Output:
        Einstein Angle in microarcseconds

    """
    if len(samples.shape) > 1:
        m_vec = samples[:, 0]
        x_vec = samples[:, 1]
        ds_vec = samples[:, 2]
    else:
        m_vec, x_vec, ds_vec = samples[[0, 1, 2]]
    ds_au = (ds_vec*u.pc).to(u.au)
    pirel = ((1/x_vec)-1)*(1.*u.au/(ds_au))
    pirel = pirel*u.radian
    pirel = pirel.to(u.microarcsecond)
    thetae = 713*(10**m_vec/0.5)**(1/2)*(pirel.value/125)**(1/2)
    if len(samples.shape) > 1:
        return thetae.reshape(len(samples), 1)
    else:
        return thetae
        