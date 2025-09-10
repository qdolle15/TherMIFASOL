import gc
import os
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def interpolate_color(start_color, end_color, scalar):
    """
    Interpolates between two colors based on a scalar between 0 and 1.
    
    :param start_color: Tuple of RGB values for the start color (e.g., (1.0, 0.0, 0.0) for red).
    :param end_color: Tuple of RGB values for the end color (e.g., (0.0, 0.0, 1.0) for blue).
    :param scalar: Float value between 0 and 1 indicating interpolation factor.
    :return: Tuple of RGB values for the interpolated color.
    """
    return tuple((1 - scalar) * start + scalar * end for start, end in zip(start_color, end_color))

def interpolate_to_black(color, scalars):
    """
    Interpolates between a given color and black based on an array of scalars between 0 and 1.
    
    :param color: Tuple of RGB values for the color (e.g., (1.0, 0.5, 0.0) for orange).
    :param scalars: Array of float values between 0 and 1 indicating interpolation factors.
    :return: List of tuples of RGB values for the interpolated colors towards black.
    """
    black = np.array([0.0, 0.0, 0.0])
    color = np.array(color)
    scalars = np.array(scalars).reshape(-1, 1)  # Ensure scalars is a column vector
    
    interpolated_colors = (1 - scalars) * color + scalars * black
    return [tuple(color) for color in interpolated_colors]

def setup_contour_parameters(columns, lines, fac):
    """
    Set up contour parameters for plotting.

    Parameters:
    - columns: Number of columns in the data.
    - lines: Number of lines in the data.
    - fac: Scaling factor for the contour.

    Returns:
    - x_contour: X-axis contour values.
    - y_contour: Y-axis contour values.
    - extent: Extent of the plot.
    - xtick_pixels: X-axis tick positions.
    - ytick_pixels: Y-axis tick positions.
    """
    x_contour = np.arange(columns) * fac
    y_contour_inverted = np.arange(lines - 1, -1, -1) * fac  # Inverted for alignment with extent
    y_contour = np.arange(lines) * fac   
    extent = [0, (columns - 1) * fac, 0, (lines - 1) * fac]
    xtick_pixels = np.arange(0, columns, 80)
    ytick_pixels = np.arange(0, lines, 72)

    return x_contour, y_contour, y_contour_inverted, extent, xtick_pixels, ytick_pixels


def plot_and_save_figure(fig, save_path, file_name, dpi=100):
    plt.savefig(f"{save_path}/{file_name}.png", format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    gc.collect()


def plot_temperature_evolution(save_path, cordons_selection, limit_wake, emissivity_factor, SILLAGE_PYRO_CAMERA, boundaries_data_selection, DATA, Tliquidus, Tsolidus):

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))
    for idx, cordon in enumerate(cordons_selection):
        row, col = divmod(idx, 2)

        # Pyrometer data
        low_boundary, high_boundary = boundaries_data_selection[cordon]
        TIME_pyro = DATA[cordon]['array time']
        TEMP2C_pyro = DATA[cordon]['array MAtemp2C']

        # Camera data
        TIME_cam = SILLAGE_PYRO_CAMERA[cordon]['time']
        TEMP_cam = SILLAGE_PYRO_CAMERA[cordon]['temperature [C]']
        if cordon % 2 == 0:
            TEMP_cam = np.flip(TEMP_cam)

        # Plot temperature
        axes[row, col].scatter(TIME_pyro, TEMP2C_pyro, label='', s=2, c='k')
        axes[row, col].scatter(TIME_cam, TEMP_cam, s=2, c='gray', label='')
        # axes[row, col].scatter(TIME_cam, TEMP_cam, s=1, c=TEMP_cam, cmap='hot', vmin=450, vmax=1800, label='Camera: Pyrometer Trail')
        axes[row, col].hlines(y=Tliquidus, xmin=TIME_pyro[0], xmax=1.9, colors='b', lw=1, label=r'$T_{liq} = 1342^{\circ}C$')
        axes[row, col].hlines(y=Tsolidus, xmin=TIME_pyro[0], xmax=1.9, colors='g', lw=1, label=r'$T_{sol} = 1257^{\circ}C$')
        axes[row, col].set_ylim(ymin=450, ymax=1800)
        # axes[row, col].legend()
        axes[row, col].grid(True, linestyle=':', color='lightgray', linewidth=1, zorder=0)

        if cordon % 2 == 0:
            axes[row, col].set_xlim(xmin=0, xmax=1.9)
        else:
            axes[row, col].set_xlim(xmin=0, xmax=1.9)


    plt.subplots_adjust(right=0.85, hspace=0.4, wspace=0.3)

    # Créer des objets graphiques temporaires pour la légende
    legend_pyro = plt.Line2D([], [], color='k', marker='o', linestyle='None', markersize=3, label='Data pyrometer recording', zorder=2)
    legend_cam = plt.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=3, label='Camera data averaged over\npyrometer wake', zorder=2)
    legend_liquidus = plt.Line2D([], [], color='b', linestyle='-', label=r'$T_{liq} = 1342^{\circ}C$', zorder=2)
    legend_solidus = plt.Line2D([], [], color='g', linestyle='-', label=r'$T_{sol} = 1257^{\circ}C$', zorder=2)

    # Ajouter une légende commune au-dessus des sous-graphiques
    fig.legend(handles=[legend_pyro, legend_cam, legend_liquidus, legend_solidus], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plot_and_save_figure(fig, save_path, f"temperature_evolution_pos{str(limit_wake).replace('.', '_')}__emi{str(emissivity_factor).replace('.', '_')}")
        
def plot_emissivity_evolution(save_path, cordons_selection, DATA, fake_time, function, params_function):

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))
    for idx, cordon in enumerate(cordons_selection):
        row, col = divmod(idx, 2)

        # Pyrometer data
        TIME_pyro = DATA[cordon]['array time']
        TEMP2C_pyro = DATA[cordon]['array MAemissivite']

        # Phenomenological law
        tau = params_function[cordon]['tau']
        delay = params_function[cordon]['delay']

        # Plot temperature
        axes[row, col].scatter(TIME_pyro, TEMP2C_pyro, label='Pyrometer', s=2, c='k')
        axes[row, col].scatter(fake_time+delay, function(fake_time, tau), s=2, c='red', label='Camera: Sillage pyrometer')
        axes[row, col].set_ylim(ymin=0, ymax=1)
        
        # Ajouter une annotation pour afficher les paramètres
        annotation_text = f"tau = {tau:.2f}\ndelay = {delay:.2f}"
        axes[row, col].annotate(
            annotation_text, xy=(0.78, 0.25), xycoords='axes fraction',
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='wheat')
            )

        # axes[row, col].legend()
        if cordon % 2 == 0:
            # axes[row, col].set_xlim(xmin=0, xmax=1.9)
            axes[row, col].set_xlim(xmin=0, xmax=5)
        else:
            # axes[row, col].set_xlim(xmin=0, xmax=2.5)
            axes[row, col].set_xlim(xmin=0, xmax=5)


    plt.subplots_adjust(right=0.85, hspace=0.4, wspace=0.3)

    # Créer des objets graphiques temporaires pour la légende
    legend_pyro = plt.Line2D([], [], color='k', marker='o', linestyle='None', markersize=3, label='Données pyromètre')
    legend_cam = plt.Line2D([], [], color='r', marker='o', linestyle='None', markersize=3, label='Loi phénoménologique')

    # Ajouter une légende commune au-dessus des sous-graphiques
    fig.legend(handles=[legend_pyro, legend_cam], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plot_and_save_figure(fig, save_path, f"emissivity_evolution_")

def plot_thermal_images(save_path, limit_wake, list_thermal_images, cordon_selection, params_clip, Tsolidus, fac, COLUMNS, LINES):
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 6), sharex=True, sharey=True)

    for idx, c_cordon in enumerate(cordon_selection):
        row, col = divmod(idx, 2)

        im_thermo = list_thermal_images[idx]
        contours = measure.find_contours(im_thermo, Tsolidus)

        largest_contour = max(contours, key=len)
        largest_contour = np.array(largest_contour)
        symmetrical_contour = largest_contour.copy()
        symmetrical_contour[:, 0] = LINES - largest_contour[:, 0]

        im = axes[row, col].imshow(im_thermo, cmap='hot', extent=[0, (COLUMNS - 1) * fac, 0, (LINES - 1) * fac], vmin=500, vmax=1800)

        # Pyrometer trail
        sillage_pyro = np.zeros((LINES, COLUMNS, 4))
        lim_sup = int((19.2 - limit_wake) / 19.2 * 512)
        sillage_pyro[lim_sup:lim_sup + 22, :, :] = 1
        sillage_pyro[lim_sup:lim_sup + 22, :, 3] = 0.5
        axes[row, col].imshow(sillage_pyro, extent=[0, (COLUMNS - 1) * fac, 0, (LINES - 1) * fac], alpha=sillage_pyro[..., 3])

        axes[row, col].set_xticks(np.arange(0, 640, 160) * fac)
        axes[row, col].set_yticks(np.arange(0, 512, 72) * fac)
        axes[row, col].set_xticklabels([f"{x:.0f}" for x in np.arange(0, 640, 160) * fac])
        axes[row, col].set_yticklabels([f"{y:.1f}" for y in np.arange(0, 512, 72) * fac])

        if row == 2:
            axes[row, col].set_xlabel('[mm]')
        if col % 2 == 0:
            axes[row, col].set_ylabel('[mm]')

        crop_min = params_clip[c_cordon]['crop low']
        crop_max = params_clip[c_cordon]['crop high']
        axes[row, col].set_ylim(crop_min, crop_max)

        for contour in contours:
            axes[row, col].plot(fac * symmetrical_contour[:, 1], fac * symmetrical_contour[:, 0], color='green', linewidth=1)
        axes[row, col].tick_params(axis='both', which='both', length=0)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position (x, y, largeur, hauteur)
    
    # Ajout de la colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    norm = plt.Normalize(vmin=500, vmax=1800)  # Normalisation manuelle pour correspondre à imshow
    cbar_ax.imshow([[500, 1800]], cmap='inferno', norm=norm, aspect='auto', visible=False)  # Force la normalisation
    solidus_norm = (Tsolidus - 500) / (1800 - 500)  # Normalisation de Tsolidus entre 0 et 1
    cbar.ax.set_ylim(500, 1800)
    cbar.ax.axhline(y=solidus_norm, color='g', linewidth=3, linestyle="--")
    cbar.set_label('Température [°C]')

    plt.subplots_adjust(right=0.9, hspace=0.15, wspace=0.1)
    # plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plot_and_save_figure(fig, save_path, f"thermal_images_pos{str(limit_wake).replace('.', '_')}")

def create_csv_file(save_path:str, cordons_selection:list, DATA:dict, SILLAGE_PYRO_CAMERA:dict):
    """
    Creates CSV files for pyrometer and camera data for each selected cordon.

    :param save_path: Path where to save the CSV files.
    :param cordons_selection: List of cordons for which to create CSV files.
    :param info_camera: Dictionary containing camera data for each cordon.
    :param info_pyrometer: Dictionary containing pyrometer data for each cordon.
    """

    for cordon in cordons_selection:
        try:
            # Check for data presence
            if cordon not in DATA or cordon not in SILLAGE_PYRO_CAMERA:
                print(f"Missing data for cordon {cordon}. Skipping.")
                continue

            # Pyrometer data
            TIME_pyro = DATA[cordon]['array time']
            TEMP2C_pyro = DATA[cordon]['array MAtemp2C']

            # Camera data
            TIME_cam = SILLAGE_PYRO_CAMERA[cordon]['time']
            TEMP_cam = SILLAGE_PYRO_CAMERA[cordon]['temperature [C]']

            # Flip temperature data for even cordons
            if cordon % 2 == 0:
                TEMP_cam = np.flip(TEMP_cam)

            # Create pandas DataFrames
            data_pyro = pd.DataFrame({'time': TIME_pyro, 'temperature': TEMP2C_pyro})
            data_cam = pd.DataFrame({'time': TIME_cam, 'temperature': TEMP_cam})

            # Save to CSV files
            pyro_file_path = os.path.join(save_path, f'pyrometer_data_{cordon}.csv')
            cam_file_path = os.path.join(save_path, f'camera_data_{cordon}.csv')

            data_pyro.to_csv(pyro_file_path, index=False)
            data_cam.to_csv(cam_file_path, index=False)

            print(f"CSV files created for cordon {cordon}.")

        except KeyError as e:
            print(f"Missing key in data for cordon {cordon}: {e}")
        except Exception as e:
            print(f"Error creating CSV files for cordon {cordon}: {e}")

def plot_differences(ax, data, global_results, reference_id, x_values, ylabel, zoom_params=None):
    """
    Plot differences on the given axis with optional zoom inset.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - data: Data to plot (e.g., temperature_differences or emissivity_differences).
    - global_results: Dictionary containing global results.
    - reference_id: ID of the reference data.
    - x_values: X-axis values for plotting.
    - ylabel: Label for the y-axis.
    - zoom_params: Dictionary containing zoom parameters (xlim, ylim) for the inset.
    """
    
    for eps_id in global_results:
        val_eps = global_results[eps_id]['eps']
        col_eps = global_results[eps_id]['color']
        label = r'$\varepsilon_{init} =$' + (f'{val_eps} : reference' if eps_id == reference_id else f'{val_eps}')
        ax.plot(x_values, data[:, eps_id], 'o--', color=col_eps, label=label)

    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)

    if zoom_params:
        axins = inset_axes(ax, width="40%", height="30%", loc='upper right', borderpad=1)
        axins.set_xlim(zoom_params['xlim'])
        axins.set_ylim(zoom_params['ylim'])
        axins.tick_params(axis='both', which='major', labelsize=10)

        for eps_id in global_results:
            axins.plot(x_values, data[:, eps_id], 'o--', color=global_results[eps_id]['color'])

        axins.set_xticks(np.arange(zoom_params['xlim'][0], zoom_params['xlim'][1] + 1, 1))
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


def display_pixels(coords, image_shape=(512, 640), colors=None):
    """
    Display crosshairs on a 2D blank image at given pixel coordinates.

    Parameters
    ----------
    coords : list of tuple
        List of (i, j) coordinates to mark (row, column).
    image_shape : tuple
        Shape of the image (height, width).
    colors : list of str or None
        List of colors for each coordinate. If None, default to red/green/blue.
    """
    height, width = image_shape
    fake_img = np.ones(image_shape)

    # Set default colors if none provided
    if colors is None:
        default_colors = ['r', 'g', 'b', 'c', 'm', 'y']
        colors = default_colors[:len(coords)]

    plt.figure(figsize=(6, 5))
    plt.imshow(fake_img, cmap='gray', vmin=0, vmax=2)

    for (i, j), color in zip(coords, colors):
        plt.axhline(height - i, color=color, linestyle='--', linewidth=1)
        plt.axvline(j, color=color, linestyle='--', linewidth=1)

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.axis('off')
    plt.tight_layout()
    plt.show()