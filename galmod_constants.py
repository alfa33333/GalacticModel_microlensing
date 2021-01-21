"""
Constants for the galactic model
"""

import numpy as np
from astropy import constants as const
###########################################
# Angles of location of particular event
# This must be changed according to the
# event position
LR_ANGLE_EVENT = 357.8699
BR_ANGLE_EVENT = 2.6072
# Example value
#LR_ANGLE_EVENT = 359.56732242
#BR_ANGLE_EVENT = 3.21595828
#
#
#
############################################

############################################
#  Parameters from lightcurve fitting
############################################
#
TE_MEAN = 6.274
TE_ERROR = 0.057
#
RHO_MEAN = 0.060464435134475955
RHO_ERROR = 0.0006404327321859689
#
THETA_STAR = 16.2 #in miliarcsec
#
# Options from measured variables are:
# 'TE' if only the t_e is measured
# 'TE_RHO' if t_e and rho are measured
MEASURE_VAR = 'TE'
#
############################################

############################################
# Constants for Bayesian likelihood
############################################

NEWTON_G = const.G.value
c = const.c.value # pylint: disable=invalid-name
M_sun = const.M_sun.value # pylint: disable=invalid-name
GM_sun = const.GM_sun.value # pylint: disable=invalid-name
OMEGA0 = 2.0
k_G = np.sqrt(GM_sun/c**2) # pylint: disable=invalid-name
vc = 100  # pylint: disable=invalid-name
GALACTIC_CENTER = 8500
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


############################################
# Constants for probability approximation
############################################
KAPPA_CONST = 8.14 # in mas/M_solar

# Dispersion values are in km s^-1
# in Galactic coordinates
DISPERSION_BULGE = {'N': 100, 'E': 100}
DISPERSION_DISC = {'N': 20, 'E': 30}
#Velocities for galactic model in km s^-1
# in Galactic coordinates
VEL_ROT = 230
VEL_DISK = np.array([0, VEL_ROT - 10])
VEL_BULGE = np.array([0, 0])
VEL_EARTH = np.array([-0.80, 28.52])
VEL_SUN = np.array([7, 12]) + np.array([0, VEL_ROT])

GALROT_ANGLE = 56.78 # in degrees
