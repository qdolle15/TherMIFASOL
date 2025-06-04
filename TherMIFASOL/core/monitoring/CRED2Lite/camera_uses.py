import time
import numpy as np
import os
import sys
import cv2
from pathlib import Path

from TherMIFASOL.core.monitoring.CRED2Lite.camera_control import CRED2LiteCamera
from TherMIFASOL.core.monitoring.CRED2Lite.utils import (
    create_unique_directory, save_report, draw_temp_device,
    create_THERMAL_images_from_frames, create_RAW_images_from_frames,
    get_effective_FPS
)
from TherMIFASOL.core.variables.GlobalVariables import LINES_CRED, COLUMNS_CRED

try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path().resolve().parents[3]

def acquire_data(fps:float, it:float, gain:str, tuning:str, image_count: int, trig: bool, comment: str, working_dir: str, period_temperature_sensor: float, roi: float) -> None:
    """
    Acquire data from the CRED2Lite camera and save the images and temperature data.

    Args:
        fps (float): Frames per second.
        it (float): Integration time.
        gain (str): Gain level.
        tuning (str): Tuning option.
        image_count (int): Number of images to acquire. Use float('inf') for unlimited recording.
        trig (bool): External trigger state.
        comment (str): Comment for the acquisition.
        working_dir (str): Working directory name.
        period_temperature_sensor (float): Temperature recording interval.
        roi (float): Region of Interest as a fraction of the full frame.
    """

    path_nuc = os.path.join(BASE_DIR, 'TherMIFASOL/resources/tables/NUC/NUC_2pts_Temp550_WithNeutralDensity.npy')
    path_calibration = os.path.join(BASE_DIR, 'TherMIFASOL/resources/tables/FLUX/FLUX_IT40_deg1_NUC2.npy')

    path_workspace = create_unique_directory(os.path.join(BASE_DIR, 'results'), working_dir)
    path_array_data = create_unique_directory(path_workspace, 'data')

    with CRED2LiteCamera(fps=fps, it=it, gain=gain, tuning=tuning) as dev:
        images_acquired = 0
        dev.set_ext_synchronization(trig)

        st_line, end_line = int((1 - roi) / 2 * LINES_CRED), int((1 + roi) / 2 * LINES_CRED)
        st_col, end_col = int((1 - roi) / 2 * COLUMNS_CRED), int((1 + roi) / 2 * COLUMNS_CRED)

        if os.path.exists(path_workspace):
            with open(os.path.join(path_workspace, 'time.txt'), "w") as f1, open(os.path.join(path_workspace, 'temperature_camera.txt'), 'w') as f2:
                try:
                    start_acquisition = time.perf_counter()
                    time_temperature_sensor = time.perf_counter() - period_temperature_sensor
                    while images_acquired < image_count:
                        raw_img = dev.get_image()

                        if raw_img is not None:
                            if time.perf_counter() - time_temperature_sensor > period_temperature_sensor:
                                time_temperature_sensor = time.perf_counter()
                                temperatures_camera = dev._get_camera_temperature()
                                extended_temperatures_camera = temperatures_camera + (time_temperature_sensor - start_acquisition,)
                                f2.write(' '.join(map(str, extended_temperatures_camera)) + '\n')

                            np.save(os.path.join(path_array_data, f"im_{images_acquired}.npy"), raw_img)
                            f1.write(str(time.perf_counter() - start_acquisition) + '\n')
                            images_acquired += 1

                            target = np.round(np.mean(raw_img[st_line:end_line, st_col:end_col]), 2)
                            info = f"Frames saved - {images_acquired:05d}"
                            
                            sys.stdout.write("\033[F\033[K" + info + '\n')
                            sys.stdout.flush()

                except KeyboardInterrupt:
                    print(f"Acquisition stopped. Number of frames: {images_acquired}")
                else:
                    print(f"Acquisition completed. Number of frames: {images_acquired}")

        else:
            print("Check workspace path...")

        dev.set_ext_synchronization(False)
        save_report(path_workspace, dev, comment, image_count, trig, period_temperature_sensor, images_acquired)
        if False:
            create_RAW_images_from_frames(
                image_dir=path_array_data,
                output_dir=os.path.join(path_workspace, 'images_raw')
            )
            create_THERMAL_images_from_frames(
                image_dir=path_array_data,
                output_dir=os.path.join(path_workspace, 'images_thermal'),
                path_NUC=path_nuc,
                path_calibration=path_calibration
            )
        get_effective_FPS(os.path.join(path_workspace, 'time.txt'))
        draw_temp_device(path_workspace)

def live_visualization(fps: float, it: float, gain: str, tuning: str) -> None:
    """
    Perform live visualization of the camera feed.

    Args:
        FPS (float): Frames per second.
        IT (float): Integration time.
        gain (str): Gain level.
        tuning (str): Tuning option.
    """
    with CRED2LiteCamera(fps, it, gain, tuning) as dev:

        cv2.namedWindow('CRED2Lite', cv2.WINDOW_NORMAL)

        try:
            while True:
                raw_img = dev.get_image()
                if raw_img is not None:
                    raw_img[raw_img > 2**14] = 0
                    image_8bit = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    cv2.imshow('CRED2Lite', image_8bit)
                    cv2.waitKey(10)
                    mean_value = np.mean(raw_img[240:260, 310:330])
                    info = f"Mean DL value: {mean_value}"

                    sys.stdout.write("\033[F\033[K" + info + '\n')
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("Visualization stopped.")
        finally:
            cv2.destroyAllWindows()
