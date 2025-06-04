import cv2
import numpy as np
from TherMIFASOL.core.variables.GlobalVariables import (
    CHANNEL, SUB_WIDTH_XIQ, SUB_LENGTH_XIQ, LINES_CRED, COLUMNS_CRED,
    LAMBDA_NIR, LAMBEQ_XIMEA, FTEQ_XIMEA, QE_NIR, FWHM_NIR, C1, C2
)

def load_full_frame_nir(frame_path:str):
    """
    Load the full NIR frame for a specific camera and recording.

    Parameters:
        path (str): Path of the frame to load

    Returns:
        np.array: Full NIR frame.
    """
    full_frame = np.load(frame_path)
    return full_frame

def load_splitted_channels_nir(frame_path):
    """
    Load NIR data for a specific camera and recording.

    Parameters:
        path (str): Path of the frame to load

    Returns:
        np.array: Loaded NIR data with split channels (LINES, COLUMNS, CHANNELS)
    """
    
    full_frame = np.load(frame_path)[:-3, :-3]  # Remove the three last columns and lines for a same size in the sub frames

    # Organize channels
    img_nir=np.zeros((CHANNEL, SUB_WIDTH_XIQ, SUB_LENGTH_XIQ))
    period_channel = np.sqrt(CHANNEL)
    for cpt in range(CHANNEL):
        i = int(cpt // period_channel)
        j = int(cpt % period_channel)
        img_nir[cpt, :, :] = full_frame[i::period_channel, j::period_channel]

    return img_nir

def split_channel(img_to_split):
    """
    Split the full frame into individual channels.

    Parameters:
    img_to_split (np.array): Full frame data.

    Returns:
    np.array: Array of split channels.
    """
    img_to_split=img_to_split[:-3,:-3]
    img_splitted = np.zeros((CHANNEL, SUB_WIDTH_XIQ, SUB_LENGTH_XIQ))

    period_channel = int(np.sqrt(CHANNEL))
    for cpt in range(CHANNEL):
        i = int(cpt // period_channel)
        j = int(cpt % period_channel)
        img_splitted[cpt, :, :] = img_to_split[i::period_channel, j::period_channel]

    return img_splitted

def xiq_dl_to_flux_two_table(full_scale_array, nuc_table, calibration_table_interpolated, calibration_table_extrapolate):
    """
    Convert DL data to flux using two tables.

    Parameters:
        full_array (np.array): Array of DL data.
        nuc_table (np.array): Array for non uniformity correction on the full frame image (2, 1088, 2048)
        calibration_table_interpolated (np.array): Array for calibration table for flux corresponding to a black body temperature below 1200°C
        calibration_table_extrapolate: Array for calibration table for flux corresponding to a black body temperature above 1200°C

    Returns:
        tuple: Two arrays (1088, 2048) - Extrapolated and interpolate flux.
    """

    dl_corrected = full_scale_array * nuc_table[0] + nuc_table[1]

    flux_interpolate = (calibration_table_interpolated[0] * dl_corrected**5 + 
                        calibration_table_interpolated[1] * dl_corrected**4 + 
                        calibration_table_interpolated[2] * dl_corrected**3 + 
                        calibration_table_interpolated[3] * dl_corrected**2 + 
                        calibration_table_interpolated[4] * dl_corrected + 
                        calibration_table_interpolated[5])
    
    flux_extrapolate = dl_corrected * calibration_table_extrapolate[0] + calibration_table_extrapolate[1]

    return flux_extrapolate, flux_interpolate

def xiq_bichromatic_temperature_field(all_channels_split_flux, channel_i, channel_j):
    """
    Calculate temperature using bichromatic method.

    Parameters:
        all_channels_split_flux (np.array): Full frame for flux (25, 217, 409)
        canal_i (int): Channel identifier i.
        canal_j (int): Channel identifier j.

    Returns:
        np.array: Bichromatic calculated temperature in Kelvin
    """
    flux_channel_i, flux_channel_j = all_channels_split_flux[channel_i], all_channels_split_flux[channel_j]  # Flux for channel i and j
    keq_i, keq_j = FTEQ_XIMEA[channel_i], FTEQ_XIMEA[channel_j]  # Constant for the equivalent Wien's law
    lambda_eq_i, lambda_eq_j = LAMBEQ_XIMEA[channel_i], LAMBEQ_XIMEA[channel_j]  # Equivalent wavelength for equivalent Wien's law
    imax_i, imax_j = QE_NIR[channel_i], QE_NIR[channel_j]  # Quantum efficiency
    fwhm_i, fwhm_j = FWHM_NIR[channel_i], FWHM_NIR[channel_j]  # Full width at half maximum
    lambda_m_i, lambda_m_j = LAMBDA_NIR[channel_i], LAMBDA_NIR[channel_j]  # Maximum quantum efficiency wavelength

    sigma_i = fwhm_i / (2 * np.sqrt(2 * np.log(2)))
    qe_i = imax_i * np.exp(-0.5 * ((lambda_eq_i - lambda_m_i) / sigma_i)**2)
    sigma_j = fwhm_j / (2 * np.sqrt(2 * np.log(2)))
    qe_j = imax_j * np.exp(-0.5 * ((lambda_eq_j - lambda_m_j) / sigma_j)**2)

    return C2 * (1 / lambda_eq_j - 1 / lambda_eq_i) / (np.log(flux_channel_i / flux_channel_j * qe_j / qe_i * keq_j / keq_i * (lambda_eq_i / lambda_eq_j)**5))

def xiq_emissivity_field(all_channels_split_flux, channel, bichromatic_temperature_array):
    """
    Calculate emissivity based on bichromatic temperature.

    Parameters:
        all_channels_split_flux (np.array): Full frame for flux (25, 217, 409)
        channel (int): Channel identifier.
        bichromatic_temperature_array (np.array): Bichromatic temperature.

    Returns:
        float: Calculated emissivity.
    """
    phi_i = all_channels_split_flux[channel]  # Flux
    lambda_eq_i = LAMBEQ_XIMEA[channel]  # Equivalent wavelength for equivalent Wien's law
    keq_i = FTEQ_XIMEA[channel]  # Constant for the equivalent Wien's law# Constant for the equivalent Wien's law
    imax_i = QE_NIR[channel]  # Quantum efficiency
    fwhm_i = FWHM_NIR[channel]  # Full width at half maximum
    lambda_m_i = LAMBDA_NIR[channel]  # Maximum quantum efficiency wavelength
    sigma_i = fwhm_i / (2 * np.sqrt(2 * np.log(2)))

    num = phi_i * (lambda_eq_i)**5 * np.exp(C2 / (lambda_eq_i * bichromatic_temperature_array))
    denom = keq_i * imax_i * C1 * np.exp(-0.5 * ((lambda_eq_i - lambda_m_i) / sigma_i)**2)

    return num / denom

def apply_homographic_transformation(array_to_rescale, homographic_matrix):
    """
    Apply homographic transformation to align the image with the CRED field.

    Parameters:
        full_array (np.array): Array of dimension (217, 409).
        homographic_matrix (np.array): Array for homographic transformation (3, 3)
    
    Returns:
        np.array: Transformed image in the CRED field of view at the same size.
    """
    xiq_rescaled = cv2.warpPerspective(array_to_rescale, homographic_matrix, (COLUMNS_CRED, LINES_CRED))
    return xiq_rescaled
