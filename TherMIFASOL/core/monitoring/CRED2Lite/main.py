from TherMIFASOL.core.monitoring.CRED2Lite.camera_uses import acquire_data, live_visualization

# Parameters for data acquisition
FPS=600  # Hz
IT=42e-6  # sec
GAIN='low'
TUNING='general'

IMAGE_COUNT = float('inf')  # Set to float('inf') for unlimited recording
TRIG = False
COMMENT = 'TherMIFASOL test'
WORKING_DIR = 'test_dir'

SAVE=False
if __name__ == "__main__":

    if SAVE:
        # Run data acquisition
        acquire_data(
            fps=FPS,
            it=IT,
            gain=GAIN,
            tuning=TUNING,
            image_count=IMAGE_COUNT,
            trig=TRIG,
            comment=COMMENT,
            working_dir=WORKING_DIR,
            period_temperature_sensor=0.1,
            roi=0.3
        )
    
    else:
        # Parameters for live visualization
        FPS = 10
        IT = 2500e-6
        GAIN = "low"
        TUNING = "general"

        # Run live visualization
        live_visualization(
            fps=FPS,
            it=IT,
            gain=GAIN,
            tuning=TUNING,
        )