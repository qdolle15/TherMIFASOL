import sys
import os
import glob
import time
import numpy as np
from pathlib import Path

# SDK of CRED2Lite management
#-------------------------------------------------------------------------
# Append the path to the FliSdk_V2 library
fli_path = os.environ.get("FLI_SDK_PATH")

if fli_path:
    fli_path = Path(fli_path)
else:
    # Default path
    fli_path = Path("C:/Program Files/FirstLightImaging/FliSdk/Python/lib")

# Check and add path
if fli_path.exists():
    sys.path.append(str(fli_path.resolve()))
    try:
        import FliSdk_V2
    except ImportError as e:
        raise ImportError("FliSdk_V2 found but failed to import. Check SDK installation.") from e
else:
    raise FileNotFoundError(f"FliSdk_V2 path not found: {fli_path}")
#-------------------------------------------------------------------------

from TherMIFASOL.core.data_processing.CRED_functions import (
    load_cred_array, dl_to_nuc, dl_to_flux, dl_to_thermo
)

PATH_TABLE_NUC = "./tables/NUC/NUC_2pts_Temp550_WithNeutralDensity.npy"
PATH_TABLE_FLUX = "./tables/FLUX/FLUX_IT40_deg1_NUC2.npy"

class CRED2LiteCamera:
    """
    A class to interface with the CRED2 Lite camera using the FliSdk_V2 library.

    Attributes:
        MODES (list): Available modes for the camera.
        SENSITIVITY_LEVELS (list): Available sensitivity levels for the camera.
        TUNING_OPTIONS (list): Available tuning options for the camera.
    """

    MODES = ['raw', 'thermal', 'FLUX', 'NUC']
    SENSITIVITY_LEVELS = ['low', 'medium', 'high']
    TUNING_OPTIONS = ['short_exposure', 'general']

    def __init__(self, fps: float, it: float, gain: str, tuning: str, mode: str = 'raw') -> None:
        """
        Initialize the CRED2LiteCamera with the given parameters.

        Args:
            fps (float): Frames per second.
            it (float): Integration time.
            gain (str): Gain level.
            tuning (str): Tuning option.
            mode (str): Camera mode. Default is 'raw'.
        """
        self.camera = False
        self.num_image = 0
        self.model = None
        self.min_tint = 0
        self.max_tint = 0
        self.context = None
        self.mode = mode
        self.emissivity = 0.73
        self.fps = fps
        self.it = it
        self.gain = gain
        self.tuning = tuning
        self.bff = -1
        self.deg_nuc = None
        self.deg_flux = None
        self.table_nuc = None
        self.table_flux = None

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self._initialize_camera()
        # self._load_tables()  Further improvement for thermal imaging in real time
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

    def _initialize_camera(self) -> None:
        """Initialize the camera and set the initial parameters."""
        print("Opening device...")
        self.context = FliSdk_V2.Init()
        list_of_grabbers = FliSdk_V2.DetectGrabbers(self.context)
        list_of_cameras = FliSdk_V2.DetectCameras(self.context)

        print(list_of_cameras)

        self.model = list_of_cameras[0]
        FliSdk_V2.SetCamera(self.context, self.model)
        FliSdk_V2.SetMode(self.context, FliSdk_V2.Mode.Full)

        if FliSdk_V2.Update(self.context):
            print("Device ready...")

        self._set_gain(self.gain)
        self._set_tuning(self.tuning)
        self._set_fps(self.fps)
        self._set_it(self.it)
        print("Device opened!")

        self.camera = FliSdk_V2.Start(self.context)

    def _load_tables(self) -> None:
        """Load the NUC and FLUX tables from the specified paths."""
        self.table_nuc = np.load(PATH_TABLE_NUC)
        self.deg_nuc = self.table_nuc.shape[2]
        self.table_flux = np.load(PATH_TABLE_FLUX)
        self.deg_flux = self.table_flux.size

    def _get_gain(self) -> str | None:
        """Get the current gain level of the camera."""
        state, response = FliSdk_V2.FliSerialCamera.SendCommand(self.context, "sensitivity")
        return response.split(':')[1] if state else None

    def _get_tuning(self) -> str | None:
        """Get the current tuning option of the camera."""
        state, response = FliSdk_V2.FliSerialCamera.SendCommand(self.context, "tuning")
        return response.split(': ')[1] if state else None

    def _get_fps(self) -> float | None:
        """Get the current frames per second of the camera."""
        state, response = FliSdk_V2.FliSerialCamera.GetFps(self.context)
        return response if state else None

    def _get_it(self) -> float | None:
        """Get the current integration time of the camera."""
        state, response = FliSdk_V2.FliSerialCamera.SendCommand(self.context, "tint raw")
        return float(response) if state else None

    def _get_min_it(self) -> float | None:
        """Get the minimum integration time of the camera."""
        state, response = FliSdk_V2.FliSerialCamera.SendCommand(self.context, "mintint raw")
        return float(response) if state else None

    def _get_max_it(self) -> float | None:
        """Get the maximum integration time of the camera."""
        state, response = FliSdk_V2.FliSerialCamera.SendCommand(self.context, "maxtint raw")
        return float(response) if state else None

    def _get_incremental_it_step(self) -> float | None:
        """Get the incremental integration time step of the camera."""
        state, response = FliSdk_V2.FliSerialCamera.SendCommand(self.context, "tintstep")
        return float(response.split(': ')[1]) if state else None

    def _get_camera_temperature(self) -> tuple | None:
        """Get the temperature of the camera."""
        return FliSdk_V2.FliCredTwoLite.GetAllTemp(self.context)[1:]

    def _get_buffer_id(self) -> int:
        """Get the current buffer ID of the camera."""
        return FliSdk_V2.GetBufferFilling(self.context)

    def _set_gain(self, value: str) -> bool:
        """Set the gain level of the camera."""
        if value in self.SENSITIVITY_LEVELS:
            state, _ = FliSdk_V2.FliSerialCamera.SendCommand(self.context, f"set sensitivity {value}")
            return state
        return False

    def _set_tuning(self, value: str) -> bool:
        """Set the tuning option of the camera."""
        if value in self.TUNING_OPTIONS:
            state, _ = FliSdk_V2.FliSerialCamera.SendCommand(self.context, f"set tuning {value}")
            return state
        return False

    def _set_fps(self, value: float) -> bool:
        """Set the frames per second of the camera."""
        return FliSdk_V2.FliSerialCamera.SetFps(self.context, float(value))

    def _set_it(self, value: float) -> bool:
        """Set the integration time of the camera."""
        res, _ = FliSdk_V2.FliSerialCamera.SendCommand(self.context, f"set tint {value}")
        return res

    def set_ext_synchronization(self, state: bool) -> None:
        """Set the external synchronization state of the camera."""
        if state:
            commands = [
                "set swsynchro on",
                f"set nbframesperswtrig 4095",
                "set swsynchro source external",
                "set extsynchro source external"
            ]
        else:
            commands = [
                "set swsynchro off",
                "set swsynchro source swtrig"
            ]

        for cmd in commands:
            FliSdk_V2.FliSerialCamera.SendCommand(self.context, cmd)

        _, state_synchro = FliSdk_V2.FliSerialCamera.SendCommand(self.context, "swsynchro")
        print(f"Synchronization state: {state_synchro}")

    def get_image(self) -> np.ndarray | None:
        """Get an image from the camera and process it based on the current mode."""
        if FliSdk_V2.GetBufferFilling(self.context) != self.bff:
            raw_image = FliSdk_V2.GetRawImageAsNumpyArray(self.context, -1)
            self.bff = FliSdk_V2.GetBufferFilling(self.context)
            metadata = {
                't(s)': time.time(),
                'ImageUniqueID': self.num_image,
            }
            self.num_image += 1

            if self.mode == 'thermal' and self.deg_nuc is not None and self.deg_flux is not None:
                array_therm = dl_to_thermo(
                    dl_raw=raw_image,
                    nuc_table=self.table_nuc,
                    calibration_table=self.table_flux,
                    emissivity=self.emissivity
                )
                return array_therm
            elif self.mode == 'FLUX' and self.deg_nuc is not None and self.deg_flux is not None:
                array_flux = dl_to_flux(
                    dl_raw=raw_image,
                    nuc_table=self.table_nuc,
                    calibration_table=self.table_flux,
                )
                return array_flux
            elif self.mode == 'NUC' and self.deg_nuc is not None:
                array_nuc = dl_to_nuc(
                    dl_raw=raw_image,
                    nuc_table=self.table_nuc,
                )
                return array_nuc
            else:
                return raw_image
        return None

    def close(self) -> None:
        """Close the camera and release resources."""
        if self.camera:
            FliSdk_V2.Stop(self.context)
            FliSdk_V2.Exit(self.context)
        else:
            print("Camera is not opened or already closed.")

    def shutdown(self) -> None:
        """Shutdown the camera."""
        self.close()
        FliSdk_V2.FliSerialCamera.SendCommand(self.context, "shutdown")
