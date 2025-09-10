import numpy as np
import warnings
from typing import Union

warnings.filterwarnings("ignore", category=RuntimeWarning)

from TherMIFASOL.core.variables.BoardCardFilterVariable import (
    params_linear, params_quadratic
)

# Wagner law
def sigma_evolution_quadratic(depth, unit, parameters):
    """
    Calculate sigma for quadratic evolution based on depth and unit.

    Parameters:
    depth (float): Depth value.
    unit (str): Unit of the depth value.
    parameters (dict): Dictionary of parameters.

    Returns:
    float: Calculated sigma value.
    """    
    if unit == 'mm':
        h11 = parameters['h11mm']
        h13 = parameters['h13mm']
        h15 = parameters['h15mm']
    else:
        h11 = parameters['h11px']
        h13 = parameters['h13px']
        h15 = parameters['h15px']

    sigma11 = parameters['sigma11']
    sigma13 = parameters['sigma13']
    sigma15 = parameters['sigma15']

    a_coefficient = (-h11*sigma13 + h11*sigma15 + h13*sigma11 - h13*sigma15 - h15*sigma11 + h15*sigma13)/(h11**2*h13 - h11**2*h15 - h11*h13**2 + h11*h15**2 + h13**2*h15 - h13*h15**2)
    b_coefficient = (h11**2*sigma13 - h11**2*sigma15 - h13**2*sigma11 + h13**2*sigma15 + h15**2*sigma11 - h15**2*sigma13)/(h11**2*h13 - h11**2*h15 - h11*h13**2 + h11*h15**2 + h13**2*h15 - h13*h15**2)
    c_coefficient = (h11**2*h13*sigma15 - h11**2*h15*sigma13 - h11*h13**2*sigma15 + h11*h15**2*sigma13 + h13**2*h15*sigma11 - h13*h15**2*sigma11)/(h11**2*h13 - h11**2*h15 - h11*h13**2 + h11*h15**2 + h13**2*h15 - h13*h15**2)

    return a_coefficient*depth**2 + b_coefficient * depth + c_coefficient

def sigma_evolution_linear(prof, mode, parameters):
    """
    Calculate sigma for linear evolution based on depth and unit.

    Parameters:
    depth (float): Depth value.
    unit (str): Unit of the depth value ('mm' or 'px').
    parameters (dict): Dictionary of parameters.

    Returns:
    float: Calculated sigma value.
    """
    
    if mode == 'mm':
        h11 = parameters['h11mm']
        h13 = parameters['h13mm']
    else:
        h11 = parameters['h11px']
        h13 = parameters['h13px']

    sigma11 = parameters['sigma11']
    sigma13 = parameters['sigma13']

    a_coefficient = (sigma13 - sigma11) / (h13 - h11)
    b_coefficient = sigma11 - a_coefficient * h11

    return a_coefficient*prof + b_coefficient

def wagner_inspired_emissivity(temperature, depth, unit, block, evolution, eps_min) -> np.ndarray:
    """
    Apply the new law to calculate the result based on temperature and other parameters.

    Parameters:
        temperature (np.ndarray): Temperature value.
        depth (np.ndarray): Depth to the ongoing bead deposition.
        unit (str): Units of the depth values ('mm' or 'px').
        apply_block (bool): Flag to determine if blocking should be applied.
        evolution_type (str): Type of evolution ('linear' or 'quadratic').
        epsilon_min (float): Minimum epsilon value.

    Returns:
        float: Calculated result based on the new law.
    """

    if evolution == 'linear':
        params = params_linear
        sigma = sigma_evolution_linear(depth, unit, params)
    else:
        params = params_quadratic
        sigma = sigma_evolution_quadratic(depth, unit, params)

    a = params['a']
    n = params['n']
    b0 = params['b0']
    T0 = params['T0']  


    temperature = np.asarray(temperature)
    result = np.full_like(temperature, b0, dtype=np.float64)
    mask = temperature > T0

    # Check if sigma is a scalar or array
    if np.isscalar(sigma):
        # Use scalar directly
        result[mask] = a * ((temperature[mask] - T0) / sigma)**(1/n) + b0
    else:
        result[mask] = a * ((temperature[mask] - T0) / sigma[mask])**(1/n) + b0

    if block:
        a_linear, b_linear = -2.84957815e-04, 8.28861586e-01
        linear_lower_bound = a_linear * temperature + b_linear
        result = np.maximum(result, np.maximum(linear_lower_bound, eps_min))
    else:
        result = np.maximum(result, eps_min)

    return result


# First order law
def time_constant_evolution(depth: float, coefficient1: float, coefficient2: float) -> float:
    """
    Calculate the evolution of tau based on depth and coefficients.

    Parameters:
    - depth (float): The depth value.
    - coef1 (float): The first coefficient for the quadratic term.
    - coef2 (float): The second coefficient for the quadratic term.

    Returns:
    - float: The calculated tau value.
    """
    tau = coefficient1 * depth**2 + coefficient2
    return tau

def first_order_emissivity(time: np.ndarray, depth: float, delay: float, params_tau_evolve: tuple[float, float], eps_non_oxy: float = 0.23, eps_oxy: float = 0.73) -> np.ndarray:
    """
    Calculate a first-order response based on time, depth, and delay.

    Parameters:
    - time (np.ndarray): Array of time values.
    - depth (float): The depth value.
    - delay (float): The delay value.
    - eps_non_oxy (float): The non-oxidized emissivity coefficient. Default is 0.23.
    - eps_oxy (float): The oxidized emissivity coefficient. Default is 0.73.
    - params_tau_evolve (tuple[float, float]): Parameters for tau evolution.

    Returns:
    - np.ndarray: The calculated response values.
    """
    tau = time_constant_evolution(depth, *params_tau_evolve)
    response = np.where(
        time < delay,
        eps_non_oxy,
        eps_non_oxy + (eps_oxy - eps_non_oxy) * (1 - np.exp(-(time - delay) / tau))
    )
    return response

# tools
def mean_absolute_image_difference(im1: Union[np.ndarray, None], im2: Union[np.ndarray, None]) -> float:
    """
    Calculate the average absolute difference between two images, ignoring NaN values.

    Parameters:
    im1 (np.ndarray): The first image as a NumPy array.
    im2 (np.ndarray): The second image as a NumPy array.

    Returns:
    float: The average absolute difference between the two images.
    """
    if im1 is None or im2 is None:
        raise ValueError("Input images must not be None.")

    if im1.shape != im2.shape:
        raise ValueError("Input images must have the same shape.")

    # Create masks for non-NaN values
    mask_im1 = ~np.isnan(im1)
    mask_im2 = ~np.isnan(im2)
    mask_both = mask_im1 & mask_im2

    # Check if there are any valid overlapping pixels
    if not np.any(mask_both):
        raise ValueError("No valid overlapping pixels between the two images.")

    # Calculate the absolute difference and the average
    abs_diff = np.abs(im1[mask_both] - im2[mask_both])
    return np.sum(abs_diff) / np.sum(mask_both)
