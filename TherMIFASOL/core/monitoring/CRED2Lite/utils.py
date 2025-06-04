import os
import re
import gc
import time
import numpy as np
from tqdm import tqdm
from skimage import measure
import matplotlib.pyplot as plt

from TherMIFASOL.core.variables.GlobalVariables import COLUMNS_CRED, LINES_CRED, FAC, T_SOLIDUS
from TherMIFASOL.core.data_processing.CRED_functions import load_cred_array, dl_to_thermo

import scienceplots
plt.style.use(['science', 'notebook', 'grid'])

def create_unique_directory(base_dir: str, dir_name: str) -> str:
    """
    Create a unique directory by appending a suffix if the directory already exists.

    Args:
        base_dir (str): Base directory path.
        dir_name (str): Desired directory name.

    Returns:
        str: Path to the created directory.
    """
    if dir_name in os.listdir(base_dir):
        extra = time.strftime("%Y%m%d-%H%M%S")
        dir_name = f"{dir_name}_{extra}"

    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def save_report(dir_path: str, cred_instance, comment: str, image_count: int, trig: bool, temp_period: float, images_saved: int) -> None:
    """
    Save acquisition details to a report file.

    Args:
        dir_path (str): Path to the directory where the report will be saved.
        cred_instance (CRED2LiteCamera): Instance of the CRED2LiteCamera.
        comment (str): Comment for the acquisition.
        image_count (int): Number of images to acquire.
        trig (bool): External trigger state.
        temp_period (float): Temperature recording interval.
    """
    with open(os.path.join(dir_path, 'report.txt'), 'w') as file:
        file.write(time.ctime(time.time()))
        file.write(f"\nComment: {comment}\n")
        file.write(f"\nNumber of images to acquire: {image_count if image_count != float('inf') else 'Unlimited'}")
        file.write(f"\nImages saved: {images_saved}")
        file.write(f"\nCamera: {cred_instance.model}\nGain: {cred_instance._get_gain()}\nTuning: {cred_instance._get_tuning()}")
        file.write(f"\nExpected FPS: {cred_instance.fps}\nActual FPS: {cred_instance._get_fps()}")
        file.write(f"\nExpected IT: {cred_instance.it}\nActual IT: {cred_instance._get_it()} sec")
        file.write(f"\nExternal Trigger: {trig}")
        file.write(f"\nCamera temperature recording interval: {temp_period} sec")

def draw_temp_device(dir_path: str) -> None:
    """
    Plot and save the internal temperatures of the camera.

    Args:
        dir_path (str): Path to the directory where the plot will be saved.
    """
    path_workspace = os.path.join(dir_path, 'workspace')
    T_CPU, T_backend, T_interface, T_ambient, T_sensor, time_ = np.loadtxt(
        os.path.join(path_workspace, 'temperature_camera.txt')).T

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(time_, T_CPU, 'o--', c='r', lw=2, label='CPU')
    ax.plot(time_, T_backend, 'o--', c='g', lw=2, label='Backend')
    ax.plot(time_, T_interface, 'o--', c='b', lw=2, label='Interface')
    ax.plot(time_, T_ambient, 'o--', c='k', lw=2, label='Ambient')
    ax.plot(time_, T_sensor, 'o--', c='m', lw=2, label='Sensor')

    ax.set_ylabel('Temperature [°C]')
    ax.set_xlabel('Time [sec]')
    ax.set_ylim(0, 70)
    ax.legend()
    ax.set_title('Internal Temperatures of the Camera')

    plt.tight_layout()
    plt.savefig(os.path.join(path_workspace, 'camera_temperatures.pdf'))

def natural_sort_key(s):
    """
    Key function for natural sorting of strings containing numbers.

    Args:
        s (str): String to sort.

    Returns:
        list: List of strings and integers for natural sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_and_save_figure(fig, save_path, file_name, dpi=100):
    """
    Save the figure as a PDF file.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        save_path (str): The directory to save the figure.
        file_name (str): The name of the file.
        dpi (int): The resolution of the saved figure.
    """
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{file_name}.pdf"), format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    gc.collect()

def setup_contour_parameters(columns, lines, fac):
    """
    Setup contour parameters.

    Args:
        columns (int): Number of columns in the image.
        lines (int): Number of lines in the image.
        fac (float): Scaling factor.

    Returns:
        tuple: Contour and extent setup parameters.
    """
    x_contour = np.arange(0, columns, 1) * fac
    y_contour = np.arange(0, lines, 1) * fac
    y_contour_inverted = y_contour[::-1]
    extent = [0, (columns - 1) * fac, 0, (lines - 1) * fac]
    xtick_pixels = np.arange(0, columns, 160) * fac
    ytick_pixels = np.arange(0, lines, 72) * fac
    return x_contour, y_contour, y_contour_inverted, extent, xtick_pixels, ytick_pixels

def create_RAW_images_from_frames(image_dir: str, output_dir: str) -> None:
    """
    Create a series of PNG images from a series of raw images with a colorbar, title, and additional information.

    Args:
        image_dir (str): Directory containing the raw images.
        output_dir (str): Directory to save the output PNG images.
    """
    # Get the list of image files and sort them naturally
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')], key=natural_sort_key)

    if not image_files:
        print("No images found in the directory.")
        return

    first_image = os.path.join(image_dir, image_files[0])

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Contour and extent setup
    x_contour, y_contour, y_contour_inverted, extent, xtick_pixels, ytick_pixels = setup_contour_parameters(COLUMNS_CRED, LINES_CRED, FAC)
    X, Y = np.meshgrid(np.arange(COLUMNS_CRED), np.arange(LINES_CRED))

    fig, ax = plt.subplots(figsize=(12, 6))

    image = ax.imshow(np.load(first_image), vmin=0, vmax=2**14, cmap='cividis', extent=extent)

    ax.set_xticks(xtick_pixels)
    ax.set_yticks(ytick_pixels)
    ax.set_xticklabels([f"{x:.0f}" for x in xtick_pixels])
    ax.set_yticklabels([f"{y:.1f}" for y in ytick_pixels])

    ax.set_xlabel('[mm]')
    ax.set_ylabel('[mm]')

    cbar = fig.colorbar(image, orientation='horizontal', shrink=1, aspect=50, pad=0.15)
    cbar.set_label('Digital Levels')
    
    for image_file in tqdm(image_files, desc="Raw images"):

        idx = natural_sort_key(image_file)[1]
        image_path = os.path.join(image_dir, image_file)
        frame = np.load(image_path)

        # Clear the previous plot
        image.set_data(frame)        

        # Save the figure as a PNG image
        output_image_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        plt.savefig(output_image_path, format="png", dpi=300, bbox_inches="tight")

    # Close the figure to free up memory
    plt.close(fig)
    gc.collect()

    print(f"Images saved to {output_dir}")

def create_THERMAL_images_from_frames(image_dir: str, output_dir: str, path_NUC:str, path_calibration: str) -> None:
    """
    Create a series of PNG images from a series of raw images with a colorbar, title, and additional information.

    Args:
        image_dir (str): Directory containing the raw images.
        output_dir (str): Directory to save the output PNG images.
        path_NUC (str): Path of the non-uniformity correction table.
        path_calibration (str): Path of the calibration table.
    """

    # Imports tables
    array_NUC = np.load(path_NUC)
    array_calibration = np.load(path_calibration)

    # Get the list of image files and sort them naturally
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')], key=natural_sort_key)

    # if not image_files:
    #     print("No images found in the directory.")
    #     return

    first_image = os.path.join(image_dir, image_files[0])
    contour_lines = []  # to keep track of plotted contour lines

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Contour and extent setup
    x_contour, y_contour, y_contour_inverted, extent, xtick_pixels, ytick_pixels = setup_contour_parameters(COLUMNS_CRED, LINES_CRED, FAC)
    X, Y = np.meshgrid(np.arange(COLUMNS_CRED), np.arange(LINES_CRED))

    fig, ax = plt.subplots(figsize=(12, 6))

    image = ax.imshow(load_cred_array(first_image), vmin=500, vmax=2000, cmap='hot', extent=extent)

    ax.set_xlabel('[mm]')
    ax.set_ylabel('[mm]')

    # Définition des ticks factices
    ax.set_xticks(xtick_pixels)
    ax.set_yticks(ytick_pixels)
    ax.set_xticklabels([f"{x:.0f}" for x in xtick_pixels])
    ax.set_yticklabels([f"{y:.1f}" for y in ytick_pixels])

    cbar = fig.colorbar(image, orientation='horizontal', shrink=1, aspect=50, pad=0.15)
    cbar.set_label('Temperature [°C]')
    cbar.ax.axvline(T_SOLIDUS-273.15, color='green', linewidth=2, linestyle='-')
    
    for image_file in tqdm(image_files, desc="Thermal images"):
        idx = natural_sort_key(image_file)[1]
        image_path = os.path.join(image_dir, image_file)
        frame = load_cred_array(image_path)

        thermal_image_uniform_emissivity_field = dl_to_thermo(
            dl_raw=frame,
            nuc_table=array_NUC,
            calibration_table=array_calibration,
            emissivity=0.73
        )

        # Update the image data
        image.set_data(thermal_image_uniform_emissivity_field - 273.15)

        # Remove previously plotted contours
        for line in contour_lines:
            line.remove()
        contour_lines.clear()

        # Find and plot new contours
        contours = measure.find_contours(thermal_image_uniform_emissivity_field, T_SOLIDUS)
        for contour in contours:
            line, = ax.plot(contour[:, 1]*FAC, contour[:, 0]*FAC, color='green', linewidth=2)
            contour_lines.append(line)

        # Save the figure as a PNG image
        output_image_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        plt.savefig(output_image_path, format="png", dpi=300, bbox_inches="tight")

    # Close the figure to free up memory
    plt.close(fig)
    gc.collect()

    print(f"Images saved to {output_dir}")

def get_effective_FPS(path_time_acquisition:str):

    time_acquisition = np.loadtxt(path_time_acquisition)
    No_acquisition = len(time_acquisition)
    duration = time_acquisition[-1] - time_acquisition[0]
    print(f"Fréquence efficitve de : {No_acquisition/duration} Hz, sur {No_acquisition} images")