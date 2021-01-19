"""
Constants for the galactic model
"""

import numpy as np

###########################################
# Angles of location of particular event
# This must be changed according to the
# event position
LR_ANGLE_EVENT = 357.8699
BR_ANGLE_EVENT = 2.6072
#
#
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
