import numpy as np
import pandas as pd
from datetime import datetime
from TherMIFASOL.core.variables.GlobalVariables import (
    C2, FTEQ_PYRO_1, FTEQ_PYRO_2, LEQ_PYRO_1, LEQ_PYRO_2, _K
)
from TherMIFASOL.core.models.ThermalLaws import wien_law

# Thermography laws
def pyrometer_bichromatic_temperature(temperature_monochromatic_channel_1, temperature_monochromatic_channel_2):
    """
    Calculate the bichromatic temperature.

    Parameters:
    temperature_monochromatic_channel_1 (float): Temperature monochromatic recorder by pyrometer in Celsius on channel 1 [1.60 - 1.80] µm
    temperature_monochromatic_channel_2 (float): Temperature monochromatic recorder by pyrometer in Celsius on channel 2 [1.45 - 1.60] µm

    Returns:
    float: Bichromatic temperature in Celsius.
    """
    flux_pyrometer_channel_1 = FTEQ_PYRO_1 * wien_law(LEQ_PYRO_1, temperature_monochromatic_channel_1 + 273.15)
    flux_pyrometer_channel_2 = FTEQ_PYRO_2 * wien_law(LEQ_PYRO_2, temperature_monochromatic_channel_2 + 273.15)

    numerator = ((LEQ_PYRO_2 - LEQ_PYRO_1) * C2) / (LEQ_PYRO_1 * LEQ_PYRO_2)
    denominator = np.log(
        (flux_pyrometer_channel_2 * FTEQ_PYRO_1 * LEQ_PYRO_2**5) / (flux_pyrometer_channel_1 * FTEQ_PYRO_2 * _K * LEQ_PYRO_1**5)
        )
    
    return (numerator/ denominator) - 273.15

def pyrometer_emissivity_coefficient(temperature_monochromatic, temperature_bichromatic, lambda_equivalent):
    """
    Calculate the emissivity ratio.

    Parameters:
    temperature_monochromatic (float): Monochromatic temperature in Celsius.
    temperature_bichromatic (float): Bichromatic temperature in Celsius.
    lambda_equivalent (float): Wavelength for equivalent Wien's method

    Returns:
    float: Emissivity ratio.
    """
    emissivity_coefficient = wien_law(lambda_equivalent, temperature_monochromatic + 273.15) / wien_law(lambda_equivalent, temperature_bichromatic + 273.15)

    return emissivity_coefficient

# Collect data
def get_seconds(time):
    """
    Convert time string to total seconds.

    Parameters:
    time (str): Time string in the format 'HH:MM:SS,nanoseconds'.

    Returns:
    float: Total seconds.
    """
    h_m_s, nano = time.split(',')
    h, m, s = h_m_s.split(':')
    return (int(h) * 3600 + int(m) * 60 + int(s)) + np.round(int(nano) * 1e-6, 6)

def collect_data_pyrometer(csv_path):
    """
    Collect and process pyrometer data.

    Parameters:
    csv_path (str): Path and file name to the data directory.

    Returns:
    dict: Dictionary containing processed pyrometer data.
    """
    data = pd.read_csv(csv_path, encoding='unicode_escape', sep=';')

    nbr_data = len(data.iloc[:, 0])
    date = data.iloc[0, 0]
    t_debut = data.iloc[0, 1]
    t_fin = data.iloc[nbr_data - 1, 1]

    time_1, micros_1 = t_debut.split(',')
    time_2, micros_2 = t_fin.split(',')

    delta_secondes = (datetime.strptime(time_2, "%H:%M:%S") - datetime.strptime(time_1, "%H:%M:%S")).total_seconds()
    delta_microseconds = (float(micros_2) - float(micros_1)) * 1e-6
    temps_acquisition = delta_secondes + delta_microseconds
    temps = np.linspace(0, temps_acquisition, nbr_data)

    time_ = data['Zeit']
    abs_time=np.asarray([(datetime.strptime(t__, '%H:%M:%S,%f')- datetime(1900, 1, 1)).total_seconds() for t__ in time_])
    real_time = np.asarray(list(map(get_seconds, time_)))
    real_time -= real_time[0]
    period_acquisition = real_time[1:] - real_time[:-1]
    fps_mean = np.mean(1 / period_acquisition)
    fps_std = np.std(1 / period_acquisition)

    temp_device = np.asarray([str_temp.replace(',','.') for str_temp in data['Gerätetemp']], dtype=float)
    temp_2C = np.asarray([str_temp.replace(',', '.') for str_temp in data['Q-Temp']], dtype=float)
    K1_temp = np.asarray([str_temp.replace(',', '.') for str_temp in data['K1-Temp']], dtype=float)
    K2_temp = np.asarray([str_temp.replace(',', '.') for str_temp in data['K2-Temp']], dtype=float)

    my_temp_2C = pyrometer_bichromatic_temperature(K1_temp, K2_temp)
    my_emissivity = pyrometer_emissivity_coefficient(K1_temp, my_temp_2C, LEQ_PYRO_1)

    info = {
        'date': date,
        'nb data': nbr_data,
        'debut acquisition': t_debut,
        'fin acquisition': t_fin,
        'temps acquisition': temps_acquisition,
        'FPS mean': fps_mean,
        'FPS std': fps_std,
        'array absolute time': abs_time,
        'array time': real_time,
        'array temp2C': temp_2C,
        'array tempK1': K1_temp,
        'array tempK2': K2_temp,
        'array MAtemp2C': my_temp_2C,
        'array MAemissivity': my_emissivity,
        'array tempDevice': temp_device,
    }

    return info

# Quick statistics from bead recording
def show_number_of_data(dictionary):
    """
    Print the number of measurements.

    Parameters:
    dictionary (dict): Dictionary containing measurement data.
    """
    print(f"Number of data : {dictionary['nb data']}")

def show_fps(dictionary):
    """
    Print the average FPS.

    Parameters:
    dictionary (dict): Dictionary containing FPS data.
    """
    print(f"mean FPS : {dictionary['FPS mean']} +- {dictionary['FPS std']} Hz")

def show_acquisition_time(dictionary):
    """
    Print the acquisition time.

    Parameters:
    dictionary (dict): Dictionary containing acquisition time data.
    """
    print(f"Acquisition time : {dictionary['temps acquisition']} sec")

def show_acquisition_start(dictionary):
    """
    Print the acquisition start time.

    Parameters:
    dictionary (dict): Dictionary containing start time data.
    """
    print(f"Start time acquisition : {dictionary['debut acquisition']}")

def show_acquisition_end(dictionary):
    """
    Print the acquisition end time.

    Parameters:
    dictionary (dict): Dictionary containing end time data.
    """
    print(f"End time acquisition : {dictionary['fin acquisition']}")


# vertically shift mask position
def shift_overlay_pyrometer_position(mask_to_shift, pixel_to_shift):
    """
    Vertically shift the position of an RGBA mask.

    This function shifts the spatial dimensions (height) of an RGBA mask
    by a specified number of pixels. The color channels remain unaffected.

    Parameters:
    mask_to_shift (np.array): The original RGBA mask to be shifted.
    pixel_to_shift (int): The number of pixels to shift the mask. A positive value shifts down,
                          a negative value shifts up.

    Returns:
    np.array: The shifted RGBA mask.
    """
    shifted_mask = np.zeros_like(mask_to_shift)
    for i in range(4):  # Iterate over each RGBA channel
        shifted_mask[:, :, i] = np.roll(mask_to_shift[:, :, i], shift=pixel_to_shift, axis=0)
    
    if pixel_to_shift > 0:
        shifted_mask[:pixel_to_shift, :, :] = 0
    else:
        shifted_mask[pixel_to_shift:, :, :] = 0
    
    return shifted_mask

def shift_position_pyrometer_mask(mask_to_shift, pixel_to_shift):
    """
    Shift the position of a binary mask vertically.

    This function shifts the spatial dimensions (height) of a binary mask
    by a specified number of pixels.

    Parameters:
    mask_to_shift (np.array): The original binary mask to be shifted.
    pixel_to_shift (int): The number of pixels to shift the mask. A positive value shifts down,
                          a negative value shifts up.

    Returns:
    np.array: The shifted binary mask.
    """
    shifted_mask = np.roll(mask_to_shift, shift=pixel_to_shift, axis=0)
    if pixel_to_shift > 0:
        shifted_mask[:pixel_to_shift, :] = False
    else:
        shifted_mask[pixel_to_shift:, :] = False

    return shifted_mask