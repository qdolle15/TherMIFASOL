import re
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----
# Shape
# -----
def format_storing_shape_dimension(index, value1, value2, value3, value4) -> str:
    """
    Format a row representing shape dimensions for display or storage.

    Parameters
    ----------
        index (int or str): Identifier of the shape (e.g., row index or label).
        value1 (float): Length of the liquidus approximation.
        value2 (float): Depth of the liquidus approximation.
        value3 (float): Length of the solidus approximation.
        value4 (float): Depth of the solidus approximation.

    Returns
    -------
        (str): A formatted string with fixed-width columns and two decimal places for the numerical values.
    """

    return f"{index:^10} | {value1:^20.2f} | {value2:^20.2f} | {value3:^20.2f} | {value4:^20.2f}\n"

def shape_quantification(x_pts: np.ndarray, y_pts: np.ndarray, xc: float, yc: float) -> tuple:
    """
    Fit an ellipse to the given points with a specified center.

    Parameters:
    x_pts (np.ndarray): Array of x-coordinates of the points.
    y_pts (np.ndarray): Array of y-coordinates of the points.
    xc (float): x-coordinate of the center of the ellipse.
    yc (float): y-coordinate of the center of the ellipse.

    Returns:
    tuple: The semi-major axis (a) and semi-minor axis (b) of the fitted ellipse.
    """
    if len(x_pts) == 0 or len(y_pts) == 0:
        return -1, -1

    # Calculate the design matrix M
    M = np.vstack(((x_pts - xc)**2, (y_pts - yc)**2)).T

    # Calculate the target vector Mt1
    Mt1 = np.array([
        np.sum((x_pts - xc)**2),
        np.sum((y_pts - yc)**2)
    ])

    # Solve the linear least squares problem
    # A, B = np.linalg.lstsq(M, Mt1, rcond=None)[0]
    try:
        A, B = np.dot(np.linalg.inv(np.dot(M.T, M)), Mt1)
    except np.linalg.LinAlgError:
        A, B = -1, -1

    # Calculate the semi-major and semi-minor axes
    semi_major_axis = np.sqrt(1 / A)
    semi_minor_axis = np.sqrt(1 / B)

    return semi_major_axis, semi_minor_axis

def get_shape_quantification_from_rapport(path):
    """
    Parse a report.txt file and extract metadata, data table, and summary stats.

    Parameters
    ----------
    path : str
        Path to the report file.
    return_dict : bool, optional
        If True, return a dictionary {metadata, data, stats}.

    Returns
    -------
    dict or pandas.DataFrame
        Dictionary with keys:
        - "metadata": dict with general info from the header
        - "data": Numpy array with numeric values
        - "stats": dict with Failure/Success/Not treated percentages
    """
    with open(path, "r") as file:
        lines = [line.rstrip() for line in file]

    metadata = {}
    data = []
    stats = {}

    # --- Extract metadata ---
    for line in lines:
        if line.startswith("epsilon"):
            metadata["epsilon"] = float(line.split(":")[1].strip())
        elif line.startswith("Ellipse center"):
            coords = re.findall(r"\d+\.?\d*", line)
            metadata["ellipse_center"] = tuple(map(float, coords))
        elif line.startswith("scale factor"):
            metadata["scale_factor_px_per_mm"] = float(line.split(":")[1].split()[0])
        elif line.startswith("Solidus"):
            metadata["Tsol"] = float(line.split(":")[1].split()[0])
        elif line.startswith("Liquidus"):
            metadata["Tliq"] = float(line.split(":")[1].split()[0])

    # --- Extract table ---
    start_data = False
    for line in lines:
        if start_data:
            if line.startswith("-") or not line.strip():
                continue
            if line.startswith("Failure") or line.startswith("Success") or line.startswith("Not"):
                break
            values = [v.strip() for v in line.split("|")]
            data.append([
                int(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4])
            ])
        if line.startswith("---"):
            start_data = True

    # --- Extract stats ---
    for line in lines:
        if line.startswith("Failure"):
            stats["failure"] = int(line.split(":")[1].replace("%","").strip())
        elif line.startswith("Success"):
            stats["success"] = int(line.split(":")[1].replace("%","").strip())
        elif line.startswith("Not treated"):
            stats["not_treated"] = int(line.split(":")[1].replace("%","").strip())

    return {"metadata": metadata, "data": np.asarray(data), "stats": stats}


# --------
# Dynamics
# --------
def hunt_data_on_isotherm(working_image: np.ndarray, T_target: float,
                          reference_point: tuple, direction:bool,
                          norm_thermal_gradient: np.ndarray, orientation_thermal_gradient: np.ndarray,
                          save_path, laser_speed: float = 0.0167) -> tuple:
    """
    Extract and process thermal gradient data on an isotherm.

    Args:
        working_image (np.ndarray): The image to process.
        T_target (float): The target temperature for finding the isotherm.
        reference_point (tuple): The reference point (Xc, Yc) for filtering contour points.
        direction (bool) : Inform if the heat source is moving toward the right or toward the left.
            - True : the heat source moves toward the right.
            - False : the heat source moves toward the left.
        norm_thermal_gradient (np.ndarray): The thermal gradient norm [K/m].
        orientation_thermal_gradient (np.ndarray): The thermal gradient orientation [radians].
        save_data (bool): Whether to save the processed data to a file.
        laser_speed (float): The laser speed in m/sec - 16.7 mm/sec.

    Returns:
        tuple: A tuple containing the length, depth, isotherm indices, thermal gradient on isotherm,
               solidification speed front on isotherm, and consistent thermal data indices.
    """
    Xc, Yc = reference_point

    # Find contours
    contours = measure.find_contours(working_image, T_target)
    if not contours:
        raise ValueError("No contours found in the image.")

    largest_contour = max(contours, key=len)

    # Filter out points where x-coordinate is greater than Xc and y-coordinate is greater than Yc
    if direction:
        isotherm_coordinates = largest_contour[(largest_contour[:, 1] < Xc) & (largest_contour[:, 0] > Yc)]
    else:
        isotherm_coordinates = largest_contour[(largest_contour[:, 1] > Xc) & (largest_contour[:, 0] > Yc)]

    isotherm_indices = (isotherm_coordinates[:, 0].astype(int), isotherm_coordinates[:, 1].astype(int))

    # 1/4 ellipse approximation
    length, depth = shape_quantification(
        x_pts=isotherm_coordinates[:, 1].astype(int),
        y_pts=isotherm_coordinates[:, 0].astype(int),
        xc=Xc,
        yc=Yc
    )

    # Filter thermal gradient orientation
    if direction:
        sub_filter = np.logical_and(
            orientation_thermal_gradient[isotherm_indices] > -np.pi / 2,
            orientation_thermal_gradient[isotherm_indices] < 0
        )
    else:
        sub_filter = np.logical_and(
            (orientation_thermal_gradient[isotherm_indices] < -np.pi / 2),
            (orientation_thermal_gradient[isotherm_indices] > -np.pi)
        )

    consistent_thermal_data_indices = (
        isotherm_indices[0][sub_filter],
        isotherm_indices[1][sub_filter]
    )

    # Thermal gradient on isotherm [K/px]
    G_on_isotherm = norm_thermal_gradient[isotherm_indices][sub_filter]
    # Orientation of the thermal gradient with respect to +y [rad]
    thermal_gradient_orientation_on_isotherm = orientation_thermal_gradient[isotherm_indices][sub_filter]
    # Solidification speed front on isotherm [m/s]
    R_on_isotherm = laser_speed * np.cos(thermal_gradient_orientation_on_isotherm)

    if save_path:
        data = np.column_stack(
            (G_on_isotherm, thermal_gradient_orientation_on_isotherm, R_on_isotherm)
            )
        header = "G[K/m] Theta[rad] R[m/sec]"
        np.savetxt(save_path, data, delimiter=' ', header=header, comments='', fmt='%d')

    return length, depth, isotherm_indices, G_on_isotherm, thermal_gradient_orientation_on_isotherm, R_on_isotherm, consistent_thermal_data_indices

def classify_growth_mode(thermal_gradient_data: np.ndarray, solidification_rate_data: np.ndarray, power_fit_params: dict) -> np.ndarray:
    """
    Classify the thermal data (G, R) based on the given interpolation parameters
    for columnar and equiaxed modes.

    Args:
        thermal_gradient_data (np.ndarray): Values of the thermal gradient G.
        solidification_rate_data (np.ndarray): Values of the solidification rate R.
        power_fit_params (dict): Interpolation parameters for columnar and equiaxed modes.

    Returns:
        np.ndarray: Array containing 0 (columnar), 1 (mixed), 2 (equiaxed) for each point.
    """
    mode_list = np.zeros_like(thermal_gradient_data, dtype=int)  # 0: columnar, 1: mixed, 2: equiaxed

    # Extract the interpolation parameters
    a_col = power_fit_params['columnar']['a']
    K_col = power_fit_params['columnar']['K']

    a_equ = power_fit_params['equiaxed']['a']
    K_equ = power_fit_params['equiaxed']['K']

    # Compare each point (G_i, R_i) with the columnar and equiaxed boundaries
    for idx, (g, r) in enumerate(zip(thermal_gradient_data, solidification_rate_data)):
        # Calculate the interpolated boundaries
        G_col_boundary = K_col * r ** a_col  # G_col for the columnar boundary
        G_equ_boundary = K_equ * r ** a_equ  # G_equ for the equiaxed boundary

        # Classify according to the boundaries
        if g > G_col_boundary:
            mode_list[idx] = 0  # Columnar
        elif G_equ_boundary < g <= G_col_boundary:
            mode_list[idx] = 1  # Mixed
        else:
            mode_list[idx] = 2  # Equiaxed

    return mode_list


# --------
# Plotting
# --------
def display_hunt_criterion(R_data: np.ndarray, G_data: np.ndarray, power_fit_params: dict, save: bool, save_path_name: str) -> None:
    """
    Display the Hunt criterion plot for the given data.

    Args:
        R_data (np.ndarray): Array of solidification rates.
        G_data (np.ndarray): Array of thermal gradients.
        prof (np.ndarray): Array of profile values.
        prof_critique (float): Critical profile value for determining remelted points.
        save (bool): Whether to save the plot or display it.
        save_path_name (str): Directory and file name for saving the plot.
    """
    # Define the path to the data files
    path_data = "/media/dolle/Extreme SSD/DATA_MIFASOL_DOLLE/MIFASOL_project/data/CET_INCONEL"
    # path_data = "../../../data/CET_INCONEL/"

    # Load the columnar and equiaxed boundary data
    V_col, G_col = np.loadtxt(f"{path_data}/columnar_CET.txt").T
    V_equ, G_equ = np.loadtxt(f"{path_data}/equiaxed_CET.txt").T
    classification = classify_growth_mode(G_data, R_data, power_fit_params)

    # Créer un graphique log-log
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.loglog(V_col, G_col, label='Columnar to Equiaxed boundaries', c='k')
    ax.loglog(V_equ, G_equ, c='k')
    # Ajouter du texte en haut à gauche et en bas à droite
    ax.text(
        0.05, 0.95, 'Columnar', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', boxstyle='round,pad=0.5', alpha=0.8))
    ax.text(
        0.95, 0.05, f'Equiaxed', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', boxstyle='round,pad=0.5', alpha=0.8))

    # Plot the data points
    colors = ['g', 'b', 'r']
    for class_ in np.unique(classification):
        col = colors[class_]
        sc = ax.scatter(
            R_data[classification==class_], 
            G_data[classification==class_], 
            c=col, edgecolor='k', zorder=3
            )

    # Add labels and title
    ax.set_xlim(1e-6, 1e1)
    ax.set_ylim(1e2, 1e6)
    ax.set_xlabel('Solidification rate $V$ (m/s)', fontsize=14)
    ax.set_ylabel('Thermal gradient $G$ (K/m)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='lower left', fontsize=12)

    # Display the grid
    ax.grid(True, which="both", ls="--")

    # Display or save the plot
    if save:
        plt.savefig(save_path_name)
        plt.close(fig)
    else:
        plt.show()

def display_G_and_R_on_images(image_to_plot: np.ndarray, T_target:float,
                              hunt_coo:tuple, filtered_gradient: np.ndarray,
                              filtered_orientation: np.ndarray, crop_y_axis:tuple, saving_path:str, scale_arrow: float, laser_speed: float = 0.0167) -> None:
    """
    Display thermal gradient and solidification front velocity on images.

    Args:
        image_to_plot (np.ndarray): The image to plot.
        T_target (float): The target temperature for finding the isotherm.
        hunt_coo (tuple): The coordinates for the hunt data.
        filtered_gradient (np.ndarray): The filtered thermal gradient norm.
        filtered_orientation (np.ndarray): The filtered thermal gradient orientation.
        crop_y_axis (tuple): bottom and top limits for imshow cropping
        laser_speed (float): The laser speed in m/s.

    Returns:
        None
    """
    crop_min, crop_max = crop_y_axis
    # Set ranges for colorbars
    T_vmin, T_vmax = 500, 1800  # Temperatures [K]
    G_vmin, G_vmax = 0, 2e5  # Thermal gradients [K/mm]
    R_vmin, R_vmax = 0, laser_speed  # Solidification speed rates [m/s]

    # Calculate gradient components
    gradient_x = filtered_gradient * np.cos(filtered_orientation)
    gradient_y = filtered_gradient * np.sin(filtered_orientation)

    # Calculate front speed components
    front_speed_x = laser_speed * np.cos(filtered_orientation)
    front_speed_y = laser_speed * np.sin(filtered_orientation)

    # Normalize the front speed for color mapping
    front_speed_norm = np.sqrt(front_speed_x**2 + front_speed_y**2)
    norm_R = Normalize(vmin=R_vmin, vmax=R_vmax)

    # Create the figure with a simple layout: 1 column, 2 rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=False)
    fig.subplots_adjust(hspace=0)

    # First plot (Thermal gradient)
    cax1 = ax1.imshow(image_to_plot, cmap='gray', vmin=T_vmin, vmax=T_vmax)
    ax1.set_ylim(crop_min, crop_max)
    quiver1 = ax1.quiver(
        hunt_coo[1], hunt_coo[0],
        gradient_x, gradient_y,
        filtered_gradient,
        angles='xy', scale_units='xy', cmap='coolwarm'
    )
    ax1.scatter(hunt_coo[1], hunt_coo[0], s=1, c='g')
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Second plot (Cooling rate)
    cax2 = ax2.imshow(image_to_plot, cmap='gray', vmin=T_vmin, vmax=T_vmax)
    ax2.set_ylim(crop_min, crop_max)
    quiver2 = ax2.quiver(
        hunt_coo[1], hunt_coo[0],
        front_speed_x, front_speed_y,
        front_speed_norm,
        angles='xy', scale_units='xy', cmap='plasma'
    )
    ax2.scatter(hunt_coo[1], hunt_coo[0], s=1, c='g')
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Adjust colorbar positions
    # Vertical colorbar for temperature
    cbar_ax_temp = fig.add_axes([0.98, 0.15, 0.02, 0.6])  # [left, bottom, width, height]
    cbar_temp = fig.colorbar(cax1, cax=cbar_ax_temp, orientation='vertical')
    cbar_temp.set_label('Temperature [K]', fontsize=12)
    cbar_temp.ax.tick_params(labelsize=12)
    cbar_temp.ax.axhline(T_target, color='g', linestyle='-', linewidth=2)

    # Horizontal colorbar for thermal gradient
    cbar_ax_grad = fig.add_axes([0.15, 0.54, 0.7, 0.02])  # Close to the image
    sm_grad = ScalarMappable(norm=Normalize(vmin=G_vmin, vmax=G_vmax), cmap='coolwarm')
    cbar_grad = fig.colorbar(sm_grad, cax=cbar_ax_grad, orientation='horizontal')
    cbar_grad.set_label('Thermal Gradient [K/m]', fontsize=12)
    cbar_grad.ax.tick_params(labelsize=12)

    # Horizontal colorbar for cooling rate
    cbar_ax_rate = fig.add_axes([0.15, 0.15, 0.7, 0.02])  # Close to the image
    sm_rate = ScalarMappable(norm=norm_R, cmap='plasma')
    cbar_rate = fig.colorbar(sm_rate, cax=cbar_ax_rate, orientation='horizontal')
    cbar_rate.set_label('Solidification Front Velocity [m/s]', fontsize=12)
    cbar_rate.ax.tick_params(labelsize=12)

    ax1.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    ax2.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

    # Save the figure
    if saving_path:
        fig.savefig(saving_path, format='png', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_gradient(GradThermique, crop_thermo: np.ndarray, gradient_norm: np.ndarray, masked_X0: np.ndarray, masked_Y0: np.ndarray, masked_fx0: np.ndarray, masked_fy0: np.ndarray, masked_norm0: np.ndarray, T_target: float, dir_name: str, cpt: int) -> None:
    """
    Plot the gradient and temperature distribution.

    Args:
        GradThermique: Instance of RegularizedThermalGradient containing the gradient data.
        crop_thermo (np.ndarray): The cropped thermal image.
        gradient_norm (np.ndarray): The norm of the gradient.
        masked_X0 (np.ndarray): The x-coordinates of the masked points.
        masked_Y0 (np.ndarray): The y-coordinates of the masked points.
        masked_fx0 (np.ndarray): The x-component of the regularized gradient at masked points.
        masked_fy0 (np.ndarray): The y-component of the regularized gradient at masked points.
        masked_norm0 (np.ndarray): The norm of the gradient at masked points.
        T_target (float): The target temperature.
        dir_name (str): The directory name to save the plot.
        cpt (int): The counter for naming the saved plot.

    Returns:
        None
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Normalized Gradient range for colorbar
    norm = Normalize(vmin=gradient_norm.min(), vmax=gradient_norm.max())

    # Plot first image
    image_to_plot = np.ma.masked_where(GradThermique.mask_filled, crop_thermo)

    cax1 = ax1.imshow(image_to_plot, cmap='gray', vmin=1000, vmax=1600)

    # Extract positions where the temperature is close to the target
    xpos, ypos = np.where((image_to_plot > T_target - 5) & (image_to_plot < T_target + 5))
    ypos = ypos[xpos > 38]
    xpos = xpos[xpos > 38]
    xpos = xpos[ypos < 500]
    ypos = ypos[ypos < 500]

    ax1.scatter(ypos, xpos, s=1, c='g')

    quiver1 = ax1.quiver(
        masked_X0, masked_Y0,
        masked_fx0, masked_fy0,
        norm(masked_norm0),
        angles='xy', scale_units='xy', scale=5, cmap='coolwarm'
    )

    # Adjust layout to make space for colorbars
    fig.subplots_adjust(bottom=0.1)

    cbar_ax1 = fig.add_axes([0.15, 0.25, 0.35, 0.03])  # [left, bottom, width, height]
    cbar_ax2 = fig.add_axes([0.55, 0.25, 0.35, 0.03])  # [left, bottom, width, height]

    cbar1 = fig.colorbar(cax1, cax=cbar_ax1, orientation='horizontal')
    sm = ScalarMappable(norm=norm, cmap='coolwarm')
    cbar3 = fig.colorbar(sm, cax=cbar_ax2, orientation='horizontal')

    cbar3.set_label(r'$|\Delta T|$  [°K/mm]')
    cbar1.set_label('K [°C]')

    plt.savefig(f"{dir_name}/GRAD_TEMPOR_LONGUE_C25_{cpt}.png")
    plt.close(fig)
