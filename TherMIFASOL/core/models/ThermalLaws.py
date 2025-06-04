import numpy as np

from TherMIFASOL.core.variables.GlobalVariables import C1, C2

# Thermography laws for top hat wavelength distribution
def planck_law(wavelength, temperature):
    """
    Calculate the spectral radiance using Planck's law.

    Parameters:
    wavelength (float): Wavelength in meters.
    temperature (float): Temperature in Kelvin.

    Returns:
    float: Spectral radiance.
    """
    return (C1*wavelength**-5)*(1/(np.exp(C2/(wavelength*temperature))-1))

def wien_law(wavelength, temperature):
    """
    Calculate the spectral radiance using Wien's law.

    Parameters:
    wavelength (float): Wavelength in meters.
    temperature (float): Temperature in Kelvin.

    Returns:
    float: Spectral radiance.
    """
    return (C1 * wavelength**-5) / (np.exp(C2 / (wavelength * temperature)))

def wien_law_equivalent(temperature, FT_equivalent, wavelength_equivalent):
    """
    Calculate the spectral radiance using Wien's law equivalent.

    Parameters:
    temperature (float): Temperature in Kelvin.
    FT_equivalent (float): Coefficient to bypass integration over the spectral range.
    wavelength (float): Wavelength equivalent in meters  to bypass integration over the spectral range.

    Returns:
    float: Spectral radiance.
    """
    return FT_equivalent*(C1*wavelength_equivalent**-5)*(1/(np.exp(C2/(wavelength_equivalent*temperature))))


# Thermography laws for gaussian wavelength distribution
def wien_law_gaussian(wavelength, temperature, I_max, fwhm, center_wavelength):
    """
    Compute the spectral radiance using Wien's law with a Gaussian spectral distribution.

    Parameters:
    ----------
    wavelength : float or np.ndarray
        Wavelength(s) in meters.
    temperature : float
        Temperature in Kelvin.
    I_max : float
        Maximum intensity of the Gaussian envelope (scaling factor).
    fwhm : float
        Full width at half maximum of the Gaussian distribution (in meters).
    center_wavelength : float
        Central wavelength of the Gaussian distribution (in meters).

    Returns:
    -------
    float or np.ndarray
        Spectral radiance weighted by the Gaussian distribution.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    gaussian_weight = np.exp(-0.5 * ((wavelength - center_wavelength) / sigma) ** 2)
    wien_radiance = (C1 / wavelength**5) / np.exp(C2 / (wavelength * temperature))
    return I_max * gaussian_weight * wien_radiance

def wien_law_equivalent_gaussian(
    temperature,
    FT_equivalent,
    wavelength_eq,
    I_max,
    fwhm,
    center_wavelength
):
    """
    Compute the equivalent spectral radiance using Wien's law,
    weighted by a Gaussian distribution centered around a given wavelength.

    Parameters:
    ----------
    temperature : float
        Temperature in Kelvin.
    FT_equivalent : float
        Equivalent transmission or scaling factor (unitless).
    wavelength_eq : float
        Equivalent wavelength in meters.
    I_max : float
        Maximum intensity of the Gaussian envelope (scaling factor).
    fwhm : float
        Full width at half maximum of the Gaussian distribution (in meters).
    center_wavelength : float
        Central wavelength of the Gaussian distribution (in meters).

    Returns:
    -------
    float
        Spectral radiance weighted by the Gaussian distribution and FT_equivalent factor.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Conversion FWHM → sigma
    gaussian_weight = np.exp(-0.5 * ((wavelength_eq - center_wavelength) / sigma) ** 2)
    wien_radiance = (C1 / wavelength_eq**5) / np.exp(C2 / (wavelength_eq * temperature))
    return FT_equivalent * I_max * gaussian_weight * wien_radiance