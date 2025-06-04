import numpy as np
from TherMIFASOL.core.variables.GlobalVariables import (
    C1, C2, MAX_DL_VALUE, FTEQ_CRED, LAMBEQ_CRED
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

# def load_cred(camera_id, record_id, time_record, id_record):
#     """
#     Load CRED data from a file.

#     Parameters:
#     camera_id (int): Camera identifier.
#     record_id (int): Recording identifier.
#     time_record (list): List of recording times.
#     id_record (list): List of recording IDs.

#     Returns:
#     np.array: Raw digital level data loaded.
#     """
#     t_, id_recording = time_record[record_id], id_record[record_id]
#     arr_raw = np.load(f'./{WORK_PATH}/CRED/CRED_normal_{camera_id:05d}/{id_recording:06d}_{t_:.3f}.npy').astype(np.float64)
#     arr_raw[0, :4] = np.nan
#     return arr_raw

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


#=========================================================
#                Non-uniformity Correction
#=========================================================
def create_table_NUC(list_mean_images:list, roi_bounds:tuple, polynomial_degree:int):
    """
    Create a non-uniformity correction (NUC) table using polynomial fitting.

    This function calculates the coefficients for a polynomial fit to correct
    non-uniformity in a set of images. The correction is based on the mean
    values within a specified region of interest (ROI).

    Parameters:
    list_mean_images (list of np.array): List of mean images to be corrected.
    roi_bounds (tuple): A tuple containing (start_row, end_row, start_col, end_col)
                        defining the region of interest (ROI).
    polynomial_degree (int): The degree of the polynomial to fit.

    Returns:
    np.array: Coefficients for the polynomial fit for each pixel within the ROI.
    """
    start_row, end_row, start_col, end_col = roi_bounds
    dl_over_roi = [np.mean(img[start_row:end_row, start_col:end_col]) for img in list_mean_images]
    dl_raws = np.asarray(list_mean_images)
    num_images, rows, cols = dl_raws.shape
    table_NUC = np.zeros((rows, cols, polynomial_degree + 1))
    for row in range(rows):
        for col in range(cols):
            pixel_values = dl_raws[:, row, col]
            if np.mean(pixel_values) > 0:
                table_NUC[row, col, :] = np.polyfit(pixel_values, dl_over_roi, polynomial_degree)
    
    return table_NUC


#=========================================================
#                   Flux calibration
#=========================================================