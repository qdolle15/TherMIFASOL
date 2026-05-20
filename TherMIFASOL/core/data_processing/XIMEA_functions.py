from os import times

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
from datetime import datetime
import time

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
    period_channel = int(np.sqrt(CHANNEL))
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

#=========================================================
#                         Tools
#=========================================================
def combine_channels(channels: np.ndarray, pad: tuple = (3, 3)) -> np.ndarray:
    """
    Reconstruct a full 2D image from a 5x5 grid of downsampled image channels.

    Parameters
    ----------
    channels : np.ndarray
        Array of shape (25, H, W), each element is a low-res channel.
    pad : tuple
        Number of rows and columns to pad at the end (default: (3, 3)).

    Returns
    -------
    np.ndarray
        Full reconstructed image of shape (H*5 + pad[0], W*5 + pad[1]).
    """
    assert channels.shape[0] == 25, "Expected 25 sub-channels (5x5 grid)."
    _, H, W = channels.shape
    full_image = np.zeros((H * 5, W * 5), dtype=channels.dtype)

    for cpt in range(25):
        i, j = divmod(cpt, 5)
        full_image[i::5, j::5] = channels[cpt]

    # Restore cropped borders
    return np.pad(full_image, ((0, pad[0]), (0, pad[1])), mode='constant')

def get_IT_images(path_data_TCN: str) -> float:
    """
    Retrieve the exposure time from the metadata of a specific integration test.

    Parameters
    ----------
    path_data_TCN : str
        Path to the folder containing 'metadata.csv' and image .npy files.

    Returns
    -------
    (float) Exposure time in microseconds.

    Raises
    ------
    FileNotFoundError: If metadata.csv is missing.
    KeyError: If 'ExposureTime' is not in the metadata.
    """
    path = Path(path_data_TCN)
    metadata_file = path / 'metadata.csv'

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    metadata = pd.read_csv(metadata_file, encoding='unicode_escape', sep=',')
    
    if 'ExposureTime' not in metadata.columns:
        raise KeyError("'ExposureTime' column is missing in metadata.")

    return float(metadata['ExposureTime'].iloc[0])


#=========================================================
#                Non-uniformity Correction
#=========================================================
def select_random_image(path_data_TCN: str) -> np.ndarray:
    """
    Select a random image from a directory based on the metadata file.

    Parameters
    ----------
    path_data_TCN : str
        Path to the directory containing 'metadata.csv' and image .npy files.

    Returns
    -------
    np.ndarray
        The selected image as a 2D numpy array.

    Raises
    ------
    FileNotFoundError
        If metadata or image file is missing.
    """
    path_data_TCN = Path(path_data_TCN)
    metadata_path = path_data_TCN / 'metadata.csv'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path, encoding='unicode_escape', sep=',')

    if 't(s)' not in metadata.columns or 'ImageUniqueID' not in metadata.columns:
        raise ValueError("Metadata must contain 't(s)' and 'ImageUniqueID' columns.")

    times = metadata['t(s)'].to_numpy()
    ids = metadata['ImageUniqueID'].to_numpy()

    if len(ids) == 0:
        raise ValueError("No images listed in metadata.")

    # Choisir un index aléatoire
    idx = random.randint(0, len(ids) - 1)
    image_id = ids[idx]
    timestamp = times[idx]

    filename = path_data_TCN / f"{int(image_id):06d}_{timestamp:.3f}.npy"
    if not filename.exists():
        raise FileNotFoundError(f"Image file not found: {filename}")

    return np.load(filename)

def temporal_mean(path_data_TCN: str) -> np.ndarray:
    """
    Compute the temporal mean of image frames based on metadata.

    Parameters
    ----------
    path_data_TCN : str
        Path to the folder containing 'metadata.csv' and image .npy files.

    Returns
    -------
    np.ndarray
        2D array representing the temporal mean image.
    """
    path_data_TCN = Path(path_data_TCN)
    metadata_path = path_data_TCN / 'metadata.csv'

    npy_files = list(path_data_TCN.glob('*.npy'))
    nb_files = len(npy_files)

    # Load metadata
    metadata = pd.read_csv(metadata_path, encoding='unicode_escape', sep=',', nrows=nb_files-1)
    times = metadata['t(s)'].to_numpy()
    ids = metadata['ImageUniqueID'].to_numpy()
    nb_img = len(ids)

    frames = []

    # print(f"Loading {nb_img} frames from: {path_data_TCN}")
    #for t, i in tqdm(zip(times, ids), total=nb_img, desc="Loading frames"):
    for t, i in zip(times, ids):
        filename = path_data_TCN / f"{i:06d}_{t:.3f}.npy"
        try:
            frame = np.load(filename)
            frames.append(frame)
        except FileNotFoundError:
            print(f"⚠️  Warning: File not found {filename}")

    if not frames:
        raise RuntimeError("No frames were successfully loaded.")

    frames = np.array(frames)
    return np.mean(frames, axis=0)

def create_table_NUC_XIMEA(raw1: np.ndarray, raw2: np.ndarray,
                              ref1: np.ndarray, ref2: np.ndarray) -> np.ndarray:
    """
    Compute NUC (Non-Uniformity Correction) coefficients from two calibration points.

    Parameters
    ----------
    raw1 (np.ndarray (2D)): Temporal mean of uniform optical scene for the first record.
    raw2 (np.ndarray (2D)): Temporal mean of uniform optical scene for the second record.
    ref1 (np.ndarray (2D)): Reconstructed array of mean spatial values over the ROI for each channel of raw1.
    ref2 (np.ndarray (2D)): Reconstructed array of mean spatial values over the ROI for each channel of raw2.

    Returns
    -------
    coeff : np.ndarray
        Array of shape (2, H, W) where:
            coeff[0] = gain correction (slope),
            coeff[1] = offset correction (intercept).
    """
    assert raw1.shape == raw2.shape == ref1.shape == ref2.shape, "All input arrays must have the same shape."
    
    H, W = raw1.shape
    coeff = np.zeros((2, H, W), dtype=np.float32)

    print("Fitting polynomial for each pixel...")
    for idx in tqdm(range(H * W), desc="Pixels"):
        i, j = divmod(idx, W)

        x = [raw1[i, j], raw2[i, j]]
        y = [ref1[i, j], ref2[i, j]]
        coeff[:, i, j] = np.polyfit(x, y, deg=1)  # [slope, intercept]

    return coeff

def apply_NUC_XIMEA(image: np.ndarray, table: np.ndarray) -> np.ndarray:
    """
    Apply a 2-point NUC (gain + offset) correction to an image.

    Parameters
    ----------
    image : np.ndarray
        The raw image (2D).
    table : np.ndarray
        Correction table of shape (2, H, W): table[0] = gain, table[1] = offset.

    Returns
    -------
    np.ndarray
        Corrected image.
    """
    assert table.shape[0] == 2, "NUC table must have shape (2, H, W)."
    return image * table[0] + table[1]

#=========================================================
#                   Flux calibration
#=========================================================
def reconstruct_calibration_flows(*channels: np.ndarray, base_shape=(1085, 2045), pad=(3, 3)) -> np.ndarray:
    """
    Reconstruct full-frame calibration maps from multiple downsampled 1D inputs.

    Parameters
    ----------
    *channels : np.ndarray
        Any number of 1D arrays of shape (25,), each representing one channel (A, B, ...).
    base_shape : tuple
        Shape of the original image before padding (default: (1085, 2045)).
    pad : tuple
        Padding to apply (rows_pad, cols_pad), default is (3, 3).

    Returns
    -------
    np.ndarray
        Reconstructed array of shape (n_channels, base_shape[0] + pad[0], base_shape[1] + pad[1])
    """
    n_channels = len(channels)
    rows, cols = base_shape
    grid_size = 5
    full_frames = []

    for ch in channels:
        if len(ch) != 25:
            raise ValueError("Each channel must be a 1D array of 25 values (5x5 grid).")
        frame = np.zeros((rows, cols))
        for cpt in range(25):
            i, j = divmod(cpt, grid_size)
            frame[i::grid_size, j::grid_size] = ch[cpt]
        # Apply padding
        frame = np.pad(frame, ((0, pad[0]), (0, pad[1])), mode='constant')
        full_frames.append(frame)

    return np.stack(full_frames, axis=0)