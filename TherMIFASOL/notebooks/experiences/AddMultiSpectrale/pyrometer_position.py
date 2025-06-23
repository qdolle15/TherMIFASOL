""" Lecture des données capteurs."""
""" Code adapté pour la CRED sous crappy. """

# Imports
import csv
import os
import h5py
import cv2
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])


work_path='./position_pyro/'

#========================================================
#                     Caméra CRED
#========================================================
# Capteur
WIDTH, DEPTH = 640, 512
# im_CRED.dtype : 'uint16'


# Temps d'acquisition
metadata_CRED=pd.read_csv(f'{work_path}/CRED/metadata.csv', encoding='unicode_escape', sep=',')
tim_python_CRED=np.asarray(metadata_CRED['t(s)'])
id_im_CRED=np.asarray(metadata_CRED['ImageUniqueID'])

# Chargement image
def load_CRED(id_):
    t_, id_ =tim_python_CRED[id_], id_im_CRED[id_]
    arr_raw=np.load(f'{work_path}/CRED/{id_:06d}_{t_:.3f}.npy')
    metadata=np.zeros_like(arr_raw, dtype=bool)
    metadata[0,:4]=True
    arr = np.ma.masked_array(arr_raw, mask=metadata)
    return arr

#========================================================
#                     Caméra XIMEA
#========================================================
# Metadata
# im_XiQ_NIR : 'float64'
channel=25
sub_width=int((1088-3)//np.sqrt(channel))
sub_length=int((2048-3)//np.sqrt(channel))
metadata_NIR=pd.read_csv(f'{work_path}/XIQ/metadata.csv', encoding='unicode_escape', sep=',')
tim_python_NIR=np.asarray(metadata_NIR['t(s)'])
tim_XIMEA_NIR=np.asarray(metadata_NIR['XimeaSec']) + np.asarray(metadata_NIR['XimeaUSec'])*1e-6 
id_im_NIR=np.asarray(metadata_NIR['ImageUniqueID'])

# Chargement des matrices de transformation
with open('./MIRE/H_matrix.pkl', 'rb') as f:
    matrice_H = pickle.load(f)


# Chargement image
def load_NIR(id_):
    """ Entre 0 et 324 - 6423 pour MUR1 (essai recalage sur mur)  """
    t_, id_ =tim_python_NIR[id_], id_im_NIR[id_]
    img_NIR=np.zeros((channel, sub_width, sub_length))
    path=f"{work_path}/XIQ/{id_:06d}_{t_:.3f}.npy"
    full_frame=np.load(path)[:-3,:-3]

    # Rangement des canneaux
    for cpt in range(25):
        i=int(cpt//np.sqrt(25))
        j=int(cpt%np.sqrt(25))
        img_NIR[cpt,:,:]=full_frame[i::5,j::5]

    return img_NIR




plt.scatter(tim_python_CRED, np.arange(len(tim_python_CRED)), label='CRED')
plt.scatter(tim_python_NIR, np.arange(len(tim_python_NIR)), label='XIQ')
plt.legend()
plt.show()

# -----------------------
# Travail sur un cordon : 
# -----------------------
PERC=0.6

ref_time=min(tim_python_CRED[0], tim_python_NIR[0])
duration=tim_python_NIR[-1]-tim_python_NIR[0]
local_time_CRED=tim_python_CRED-ref_time
local_time_XIMEA=tim_python_NIR-ref_time

# Frequence plus faible on traite d'abord la XIMEA
chosen_id_XIMEA=np.abs(local_time_XIMEA - PERC*duration).argmin()
relative_time_XIMEA=local_time_XIMEA[chosen_id_XIMEA]

# Ensuite on prend l'image la plus proche de la CRED
chosen_id_CRED=np.abs(local_time_CRED - relative_time_XIMEA).argmin()
relative_time_CRED=local_time_CRED[chosen_id_CRED]


#---------------------------------------------
# Affichage du temps des images sélectionnées.
#---------------------------------------------
fig, ax = plt.subplots()

img_prise=[relative_time_CRED, relative_time_XIMEA]
extr=0.12

ax.vlines(
    img_prise,
    ymin=-0.5,
    ymax=1.5,
    color=['b', 'g'],
    linestyles='dotted'
    )
ax.scatter([0],[0], facecolor='white', edgecolor='black', label="Prise d'image")
ax.plot([0,0],[0,0], c='black', label="Images selctionnées", linestyle='dotted')

ax.scatter(local_time_CRED, np.ones(len(local_time_CRED))*0, facecolor='blue', edgecolor='black')
ax.scatter(local_time_XIMEA, np.ones(len(local_time_XIMEA))*1, facecolor='green', edgecolor='black')

ax.set_yticklabels(['', '', 'CRED','','','', 'NIR', ''])
ax.set_ylim(-0.5, 1.5)
ax.set_xlim(min(img_prise)-extr, max(img_prise)+extr)
ax.legend()
ax.set_xlabel('Temps[sec]')
plt.title(f"""Prises d'image depuis déclanchement par trig""")
plt.show()



#-------------------
# SELCTION D'IMAGES :
#-------------------
im_XiQ_NIR=load_NIR(chosen_id_XIMEA)
im_CRED=load_CRED(chosen_id_CRED)


chanel=23
XIQ_raw=im_XiQ_NIR[chanel]
H_i=matrice_H[chanel]
XIQ_rescaled=cv2.warpPerspective(XIQ_raw, H_i, (WIDTH, DEPTH))


pos_pyro=np.ones((XIQ_rescaled.shape[0], XIQ_rescaled.shape[1], 4), dtype=int)*0
pos_pyro[XIQ_rescaled==255,0]=255 
pos_pyro[XIQ_rescaled==255,3]=255

fig, axes = plt.subplots(1, 2)
cmp = 'viridis'

im1 = axes[0].imshow(im_CRED, cmap=cmp, vmin=0, vmax=2**14)
axes[0].imshow(pos_pyro)
#axes[0].set_title('CRED')
axes[0].axes.get_xaxis().set_ticks([])
axes[0].axes.get_yaxis().set_ticks([])
fig.colorbar(im1, ax=axes[0], orientation="horizontal", label="Niveaux de gris [ADU]")

im2 = axes[1].imshow(XIQ_rescaled, cmap=cmp, vmin=0, vmax=255)
axes[1].imshow(pos_pyro)
#axes[1].set_title(f'XiQ NIR')
axes[1].axes.get_xaxis().set_ticks([])
axes[1].axes.get_yaxis().set_ticks([])
fig.colorbar(im2, ax=axes[1], orientation="horizontal", label="Niveaux de gris [ADU]")

plt.tight_layout()
plt.show()

