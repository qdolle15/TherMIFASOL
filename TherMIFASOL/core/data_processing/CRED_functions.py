import warnings

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from TherMIFASOL.core.variables.GlobalVariables import (
    C1, C2, MAX_DL_VALUE, FTEQ_CRED, LAMBEQ_CRED, LINES_CRED, COLUMNS_CRED
)

def load_cred_array(path_image:str):
    """
    Load CRED data from a file.

    Parameters:
    path_image (str): Path of the numpy array.

    Returns:
    np.array: Raw digital level data loaded.
    """
    arr_raw = np.load(path_image).astype(np.float64)
    return arr_raw

def dl_to_nuc(dl_raw, nuc_table):
    """
    Apply NUC correction to raw DL data.

    Parameters:
    dl_raw (np.array): Raw DL data.
    nuc_table (np.array): Non uniformity coefficient table.

    Returns:
    np.array: NUC-corrected data.
    """
    dl_raw[dl_raw > MAX_DL_VALUE] = MAX_DL_VALUE
    dl_corrected = np.zeros_like(dl_raw)

    deg_nuc = nuc_table.shape[2]

    for i in range(deg_nuc):
        dl_corrected += nuc_table[:, :, i] * dl_raw**(deg_nuc - (i + 1))

    return dl_corrected

def dl_to_flux(dl_raw, nuc_table, calibration_table):
    """
    Convert raw DL data to flux.

    Parameters:
    dl_raw (np.array): Raw DL data.
    nuc_table (np.array): Non uniformity coefficient table.
    calibration_table (np.array): Calibration table between Digital levels and flux

    Returns:
    np.array: Flux data.
    """
    dl_raw[dl_raw > MAX_DL_VALUE] = MAX_DL_VALUE
    dl_corrected = np.zeros_like(dl_raw)
    flux = np.zeros_like(dl_raw)

    deg_nuc = nuc_table.shape[2]
    deg_flux = calibration_table.shape[0]

    for i in range(deg_nuc):
        dl_corrected += nuc_table[:, :, i] * dl_raw**(deg_nuc - (i + 1))

    for deg in range(deg_flux):
        flux += calibration_table[deg] * dl_corrected**(deg_flux - (deg + 1))

    return flux

def wien_reversed(wien_equivalent_factor, wien_equivalent_wavelength, flux):
    """
    Calculate the temperature using the reversed equivalent Wien's law.

    Parameters:
    wien_equivalent_factor (float): Equivalent transmission factor.
    wien_equivalent_wavelength (float): Wavelength.
    flux (float): Electromagnetic flux.

    Returns:
    float: Calculated temperature in Kelvin.
    """
    return C2 / (wien_equivalent_wavelength * np.log(wien_equivalent_factor * C1 * wien_equivalent_wavelength**-5 / flux))

def dl_to_thermo(dl_raw, nuc_table, calibration_table, emissivity):
    """
    Convert raw DL data to thermal data.

    Parameters:
    dl_raw (np.array): Raw Digital Levels data.
    nuc_table (np.array): Non uniformity coefficient table.
    calibration_table (np.array): Calibration table between Digital levels and flux
    emissivity (float or np.array): Emissivity coefficient / field.

    Returns:
    np.array: Thermal data.
    """

    dl_raw[dl_raw > MAX_DL_VALUE] = MAX_DL_VALUE
    dl_corrected = np.zeros_like(dl_raw)
    flux = np.zeros_like(dl_raw)

    deg_nuc = nuc_table.shape[2]
    deg_flux = calibration_table.shape[0]

    for i in range(deg_nuc):
        dl_corrected += nuc_table[:, :, i] * dl_raw**(deg_nuc - (i + 1))

    for deg in range(deg_flux):
        flux += calibration_table[deg] * dl_corrected**(deg_flux - (deg + 1))

    thermo = wien_reversed(FTEQ_CRED, LAMBEQ_CRED, flux / emissivity)
    return thermo


#=========================================================
#                  Temporal analysis
#=========================================================
def profile_ONE_pixel_correction(pixel_profile, bin_width=25):

    """
    Take the profile pixel (i.e. DL vs time) and get back normal 
    data and the associated mask.
    """

    hist, bin_edges = np.histogram(pixel_profile, bins=np.arange(
        np.min(pixel_profile), 
        np.max(pixel_profile) + bin_width,
          bin_width)
          )
    max_bin_index = np.argmax(hist)
    plateau_range = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])
    plateau_values = pixel_profile[(pixel_profile >= plateau_range[0]) & (pixel_profile < plateau_range[1])]
    plateau_mean = np.mean(plateau_values)
    mask_normal_data = (pixel_profile > plateau_mean - 3*bin_width) & (pixel_profile < plateau_mean + 3*bin_width)
    
    return pixel_profile[mask_normal_data], mask_normal_data

def focus_on(arr, pixel_ij, bin_width=25):
    """
    Visualize the temporal behavior of a pixel with statistics and correction mask.

    Parameters
    ----------
    arr : np.ndarray
        3D array of shape (time, height, width), containing digital level data.
    pixel_ij : tuple of int
        The (i, j) coordinates of the pixel to analyze.
    bin_width : int, optional
        Width of bins for the histogram, by default 25.
    """
    i, j = pixel_ij
    data = arr[:, i, j]
    time_points = np.arange(data.size)

    mean_orig = np.mean(data)
    std_orig = np.std(data)

    # 1. Plot raw temporal profile
    plt.figure(figsize=(8, 4))
    plt.plot(time_points, data, 'o--', alpha=0.5, color='blue')
    plt.axhline(mean_orig, color='blue', linestyle='--', label=fr"{mean_orig:.0f} ± {std_orig:.0f} DL")
    plt.xlabel("Frame index", fontsize=11)
    plt.ylabel("Digital Level [ADU] (14 bits)", fontsize=11)
    plt.title(f"Pixel ({i}, {j}) - Temporal Profile", fontsize=12)
    plt.xlim(-10, data.size + 10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Plot histogram and raw profile together
    hist, bin_edges = np.histogram(data, bins=np.arange(np.min(data), np.max(data) + bin_width, bin_width))

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bin_edges, orientation='horizontal', color='red', alpha=0.5, label='Histogram')
    plt.plot(time_points, data, 'o--', alpha=0.5, color='blue', label='Raw data')
    plt.xlabel("Frame index / Occurrences", fontsize=11)
    plt.ylabel("Digital Level [ADU]", fontsize=11)
    plt.title(f"Pixel ({i}, {j}) - Raw & Histogram", fontsize=12)
    plt.xlim(-10, data.size + 10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Corrected profile
    corrected_data, mask_valid = profile_ONE_pixel_correction(data)
    mean_corr = np.mean(corrected_data)
    std_corr = np.std(corrected_data)

    plt.figure(figsize=(8, 4))
    plt.plot(time_points, data, 'o--', alpha=0.5, color='blue', label='All data')
    plt.plot(time_points[mask_valid], data[mask_valid], 'o', color='red', ms=3, label='Selected data')
    plt.axhline(mean_orig, linestyle='--', color='blue', label=fr"Original: {mean_orig:.0f} ± {std_orig:.0f} DL")
    plt.axhline(mean_corr, linestyle='--', color='red', label=fr"Corrected: {mean_corr:.0f} ± {std_corr:.0f} DL")
    plt.xlabel("Frame index", fontsize=11)
    plt.ylabel("Digital Level [ADU] (14 bits)", fontsize=11)
    plt.title(f"Pixel ({i}, {j}) - Correction", fontsize=12)
    plt.xlim(-10, data.size + 10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def correct_temporal_data(DL_temporal, threshold=0.4):
    """
    Compute the average temporal digital level for each pixel,
    avoiding anomalous values based on a correction mask.

    Parameters
    ----------
    DL_temporal : np.ndarray
        3D array of shape (time, height, width) containing raw temporal data.
    threshold : float
        Threshold ratio above which a pixel is considered compromised.

    Returns
    -------
    mean_temp : np.ndarray
        2D array (height x width) of corrected mean digital levels.
    compromised_data : np.ndarray
        2D array of fraction of anomalous values for each pixel.
    problematic_pixels : list of tuple
        List of (i, j) pixels with anomaly ratio > threshold.
    """
    time_len, height, width = DL_temporal.shape
    mean_temp = np.zeros((height, width))
    compromised_data = np.zeros((height, width))
    problematic_pixels = []

    for i in range(height):
        for j in range(width):
            # Skip known corner pixels if needed
            if i == 0 and j in [0, 1, 2, 3]:
                continue

            signal = DL_temporal[:, i, j]
            corrected_signal, valid_mask = profile_ONE_pixel_correction(signal)

            ratio_anomalous = 1 - np.sum(valid_mask) / len(valid_mask)
            compromised_data[i, j] = ratio_anomalous

            if ratio_anomalous > threshold:
                problematic_pixels.append((i, j))

            mean_temp[i, j] = np.mean(corrected_signal)

    return mean_temp, compromised_data, problematic_pixels

#=========================================================
#                Non-uniformity Correction
#=========================================================
def create_table_NUC_CRED(array_temporal_mean_images:np.ndarray, roi_bounds:tuple, polynomial_degree:int):
    """
    Create a non-uniformity correction (NUC) table using polynomial fitting.

    This function calculates the coefficients for a polynomial fit to correct
    non-uniformity in a set of images. The correction is based on the mean
    values within a specified region of interest (ROI).

    Parameters:
    array_temporal_mean_images (array of np.array): List of temporal mean images to be corrected.
    roi_bounds (tuple): A tuple containing (start_row, end_row, start_col, end_col)
                        defining the region of interest (ROI).
    polynomial_degree (int): The degree of the polynomial to fit.

    Returns:
    np.array: Coefficients for the polynomial fit for each pixel within the ROI.
    """
    start_row, end_row, start_col, end_col = roi_bounds
    n_images, rows, cols = array_temporal_mean_images.shape

    # Average digital level over the ROI for each image
    dl_over_roi = np.array([
        np.mean(img[start_row:end_row, start_col:end_col])
        for img in array_temporal_mean_images
    ])

    table_NUC = np.zeros((rows, cols, polynomial_degree + 1))

    print("Fitting polynomial for each pixel...")
    for idx in tqdm(range(rows * cols), desc="Pixels"):
        row, col = divmod(idx, cols)
        pixel_values = array_temporal_mean_images[:, row, col]

        if np.mean(pixel_values) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                coeffs = np.polyfit(pixel_values, dl_over_roi, polynomial_degree)
                table_NUC[row, col, :] = coeffs

    return table_NUC


def display_nuc_analysis(
    raw_image,
    nuc_table,
    roi_bounds,
    title="NUC Correction",
    No_pts=None,
    sensor_temp_c=None
):
    """
    Display a side-by-side comparison of a raw image and its corrected version using a NUC table.

    Parameters
    ----------
    raw_image : np.ndarray or str or Path
        Raw DL image (2D array) or path to a .npy file.
    nuc_table : np.ndarray or str or Path
        NUC correction table (3D array) or path to a .npy file.
    roi_bounds : tuple
        (row_start, row_end, col_start, col_end) for ROI used to scale color limits.
    title : str
        Title of the figure.
    No_pts : int or None
        Number of points for NUC.
    sensor_temp_c : float or None
        Optional sensor temperature (displayed in title).
    """

    if isinstance(nuc_table, str):
        nuc_table = np.load(str(nuc_table))

    corrected_image = np.zeros_like(raw_image)
    degree = nuc_table.shape[2]

    # Apply NUC correction: polynomial evaluation
    for i in range(degree):
        corrected_image += nuc_table[:, :, i] * raw_image**(degree - i - 1)

    # Compute ROI-based color limits
    r1, r2, c1, c2 = roi_bounds

    roi_raw = raw_image[r1:r2, c1:c2]
    roi_corr = corrected_image[r1:r2, c1:c2]

    min_raw = np.mean(roi_raw) - 3 * np.std(roi_raw)
    max_raw = np.mean(roi_raw) + 3 * np.std(roi_raw)

    min_corr = np.mean(roi_corr) - 5 * np.std(roi_corr)
    max_corr = np.mean(roi_corr) + 5 * np.std(roi_corr)

    # Build title with metadata if provided
    full_title = title
    if No_pts is not None:
        full_title += f" — No of points: {No_pts} "
    if sensor_temp_c is not None:
        full_title += f" — Temp: {sensor_temp_c:.1f} °C"

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(17, 10))
    fig.suptitle(full_title, fontsize=14)

    # Raw image
    ax = axes[0]
    im0 = ax.imshow(raw_image, vmin=min_raw, vmax=max_raw)
    ax.set_title("Raw Image")
    plt.colorbar(im0, ax=ax, orientation='horizontal')

    # Corrected image
    ax = axes[1]
    im1 = ax.imshow(corrected_image, vmin=min_corr, vmax=max_corr)
    ax.set_title("Corrected Image (NUC)")
    plt.colorbar(im1, ax=ax, orientation='horizontal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Print std for debug or logging
    print(f"STD raw: {np.std(raw_image):.2f}")
    print(f"STD corrected: {np.std(corrected_image):.2f}")


#=========================================================
#                   Flux calibration
#=========================================================