""" functions used for fitting DLA profile in quasar spectra """

import numpy as np
from . import constants

from scipy.constants import e, epsilon_0, m_p, m_e, k
from scipy.special import voigt_profile

# need to change c from km/s to m/s
c = constants.c*1000.
# precomputed constants for voigt
GAUSSIAN_BROADENING_B = np.sqrt(k * constants.gastemp / m_p)
# 1e-10 conversion factor to Angstroms
LORENTZIAN_BROADENING_GAMMA_PREFACTOR = 1e-10 / (4 * np.pi)
TAU_PREFACTOR = (e**2 * 1e-10 / m_e / c / 4 / epsilon_0)

def compute_DLA_tau(lambda_obs, z_dla, log_nhi, lambda_t, oscillator_strength_f, gamma):
    """
    compute the optical depth of a DLA given the redshift and column density


    Arguments
    ---------
    lambda_obs (array of floats) : observed wavelength (Angstrom)
    z_dla (float) : DLA redshift
    log_nhi (float) : DLA column density in log10(cm^-2)
    lambda_t (float) : Transition wavelength (Angstrom)
    oscillator_strength_f (float) : Oscillator strength
    gamma (float) : Damping constant (s^-1)

    Return
    ------
    tau (array of floats) : optical depth at zpix, length of lambda_obs
    """

    # compute broadenings for the voight profile
    relative_velocity_nu = c * (lambda_obs / (1 + z_dla) / lambda_t - 1)
    lorentzian_broadening_gamma = (
        LORENTZIAN_BROADENING_GAMMA_PREFACTOR * gamma * lambda_t)

    # convert column density to m^2
    nhi = 10**log_nhi * 1e4

    # the 1e-10 converts the wavelength from Angstroms to meters
    tau = TAU_PREFACTOR * nhi * oscillator_strength_f * lambda_t * voigt_profile(
        relative_velocity_nu, GAUSSIAN_BROADENING_B, lorentzian_broadening_gamma)

    return(tau)


def dla_profile(lambda_obs, z_abs, nhi):
    """
    determine the mean transmission as a fucntion of observed wavelength for a DLA given
    the redshift and column density
    
    Arguments
    ---------
    lambda_obs (array of floats) : observed wavelength (Angstrom)
    z_abs (float) : DLA redshift
    nhi (float) : log10 DLA column density (cm^-2)

    Return
    ------
    T (array of floats) : mean transmission of zpix, length of lambda_obs

    """

    T = np.exp(
        -compute_DLA_tau(lambda_obs, z_abs, nhi, constants.Lya_line, constants.oscillator_strength_Lya, constants.gamma_Lya)
        -compute_DLA_tau(lambda_obs, z_abs, nhi, constants.Lyb_line, constants.oscillator_strength_Lyb, constants.gamma_Lyb)
    )

    return(T)


