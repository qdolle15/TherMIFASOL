import os
import numpy as np

# Constants related to the alloy under study
T_SOLIDUS = 1530.15  # Solidus temperature of Inconel 718 [K]
T_LIQUIDUS = 1342 + 273.15  # Liquidus temperature of Inconel 718 [K]
T_SOLIDUS_GANDIN = 1075 + 273.15  # K
T_LIQUIDUS_GANDIN = 1337 + 273.15  # K

# Physical constants
C = 299792458  # Speed of light in vacuum [m/s]
K = 1.380649e-23  # Boltzmann constant [J/K]
H = 6.62607015e-34  # Planck's constant [J·s]
C1 = 2 * H * C**2
C2 = H * C / K

# Monochromatic camera
spectral_range_monochromatic_camera = [1.1e-6, 1.7e-6]  # [m] - Take into consideration high-pass filter
LINES_CRED, COLUMNS_CRED = 512, 640  # Sensor size
FAC = 24 / 640  # Camera resolution (640 pixels <--> 24 mm)
FTEQ_CRED = 5.55137084e-07  # Constant for the equivalent Wien method ([1.1 - 1.7] µm)
LAMBEQ_CRED = 1.43174966e-06
MAX_DL_VALUE = 2**14  # Maximum Digital Level of the camera. Pixels [0,:4] are 2**16 metadata

# Bichromatic pyrometer
pyrometer_channel_1 = [1.65e-6, 1.8e-6]
pyrometer_channel_2 = [1.45e-6, 1.65e-6]
FTEQ_PYRO_1 = 1.49769538e-07  # Constant for the equivalent Wien method
FTEQ_PYRO_2 = 1.99357147e-07
LEQ_PYRO_1 = 1.72338497e-06
LEQ_PYRO_2 = 1.54772967e-06
_K = 1  # Ratio of the emissivity constants of the two channels.

## Creating a table for an RGBA image
# MASK_PYROMETER_POSITION = np.load("../position_pyro/position_pyro_640x512.npy")
# OVERLAY_MASK_POSITION_PYROMETER = np.zeros((WIDTH_CRED, DEPTH_CRED, 4))  
# OVERLAY_MASK_POSITION_PYROMETER[MASK_PYROMETER_POSITION] = [1, 0, 0, 1] 

# Multispectral camera
CHANNEL = 25  # Number of camera channels
WIDTH_XIQ, DEPTH_XIQ = 1088, 2048  # Total sensor size
SUB_WIDTH_XIQ = int((WIDTH_XIQ - 3) // np.sqrt(CHANNEL))
SUB_LENGTH_XIQ = int((DEPTH_XIQ - 3) // np.sqrt(CHANNEL))

# Channel behavior
LAMBDA_NIR = np.array([
    911.482645, 919.901124, 929.328723, 939.401237, 949.249995, 851.803589,
    862.969565, 877.450671, 889.103373, 898.081718, 787.576142, 803.275504,
    813.321367, 826.318921, 841.792014, 727.740742, 738.542621, 751.853625,
    766.684691, 779.930856, 660.365807, 668.017413, 686.229522, 699.930299,
    711.184147
]) * 1e-9

FWHM_NIR = np.array([
    15.3305785, 18.3057851, 18.3057851, 19.4214876, 17.3760331, 10.1239669,
    10.4958678, 12.5413223, 13.285124, 14.214876, 7.14876033, 7.14876033,
    8.0785124, 8.45041322, 10.1239669, 7.14876033, 8.0785124, 6.7768595,
    8.0785124, 7.14876033, 4.73140496, 6.7768595, 6.40495868, 8.45041322,
    7.14876033
]) * 1e-9

QE_NIR = np.array([
    0.0449465748, 0.0371457538, 0.0306590005, 0.0247684725, 0.0263526802,
    0.0691592659, 0.058166068, 0.0313776227, 0.034033688, 0.0438410733,
    0.0688674699, 0.103507206, 0.0906530483, 0.0561098945, 0.0538589066,
    0.148136809, 0.0956377266, 0.0969044787, 0.131132293, 0.10257433,
    0.0284376021, 0.0846240252, 0.0781149953, 0.12265537, 0.121625256
])

# Constants for the equivalent Wien method for all channels
# KEQ_OPT
FTEQ_XIMEA = np.array([
    1.63247672e-08, 1.94951542e-08, 1.94943734e-08, 2.06827215e-08,
    1.85021946e-08, 1.07794947e-08, 1.11754054e-08, 1.33541846e-08,
    1.41462526e-08, 1.51365531e-08, 7.61135458e-09, 7.61114125e-09,
    8.60132842e-09, 8.99725668e-09, 1.07797509e-08, 7.61247530e-09,
    8.60307608e-09, 7.21573346e-09, 8.60228964e-09, 7.61146875e-09,
    5.03788619e-09, 7.21776653e-09, 6.82077724e-09, 9.00113994e-09,
    7.61289932e-09
])

LAMBEQ_XIMEA = np.array([
    9.11482645e-07, 9.19901124e-07, 9.29328723e-07, 9.39401237e-07,
    9.49249995e-07, 8.51803589e-07, 8.62969565e-07, 8.77450671e-07,
    8.89103373e-07, 8.98081718e-07, 7.87576142e-07, 8.03275504e-07,
    8.13321367e-07, 8.26318921e-07, 8.41792014e-07, 7.27740742e-07,
    7.38542621e-07, 7.51853625e-07, 7.66684691e-07, 7.79930856e-07,
    6.60365807e-07, 6.68017413e-07, 6.86229522e-07, 6.99930299e-07,
    7.11184147e-07
])

# Parameters fitting the linear part within the log-log plot of the Columnar \
# to equiaxed transitions for IN718
CET_IN718_fitting_parameters = {
    'equiaxed':{
        'a':0.4967749112648437,
        'K':86866.56889917719
    },
    'columnar':{
        'a':0.46550189267556186,
        'K':323366.96068853716
    }
}