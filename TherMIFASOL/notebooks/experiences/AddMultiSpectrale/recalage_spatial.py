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


work_path='MIRE'

#========================================================
#                     Caméra CRED
#========================================================
# Capteur
WIDTH, DEPTH = 512, 640
# im_CRED.dtype : 'uint16'

# Tables
path_calibration_CRED = "../calibration/opti_SCHN_TOURM_F/tables_filtreND"
table_NUC = np.load(f'{path_calibration_CRED}/NUC_2pts_Temp550_F.npy')
pts_NUC = table_NUC.shape[2]
table_flux = np.load(f'{path_calibration_CRED}/FLUX_IT40_deg1_NUC2.npy')
pts_flux = table_flux.shape[0]

# Fonctions thermiques
c = 299792458  # m/s : vitesse de la lumière dans le vide
k = 1.380649e-23  # J/K : constante de Boltzmann
h = 6.62607015e-34  # J.s : constante de Planck
FTeq = 4.85886942e-07
lambeq = 1.34427824e-06

def WienReversed(FT_, lamb, phi):
    return h * c /(lamb * k *np.log(FT_ * 2*h*c**2*lamb**-5 / phi))

def DL_to_thermo(DL_brute, eps):
    DL_brute[DL_brute > 2**14] = 2**14
    DL_corrige = np.zeros_like(DL_brute)
    flux = np.zeros_like(DL_brute)
    for i in range(pts_NUC):
        DL_corrige += table_NUC[:,:,i] * DL_brute**(pts_NUC - (i+1))
    for deg in range(pts_flux):
        flux += table_flux[deg] * DL_corrige**(pts_flux - (deg+1))

    thermo = WienReversed(FTeq, lambeq, flux/eps)

    return thermo

def DL_to_flux(DL_brute):
    DL_brute[DL_brute > 2**14] = 2**14
    DL_corrige = np.zeros_like(DL_brute)
    flux = np.zeros_like(DL_brute)
    for i in range(pts_NUC):
        DL_corrige += table_NUC[:,:,i] * DL_brute**(pts_NUC - (i+1))
    for deg in range(pts_flux):
        flux += table_flux[deg] * DL_corrige**(pts_flux - (deg+1))

    return flux

def DL_to_flux_noNUC(DL_brute):
    DL_brute[DL_brute > 2**14] = 2**14
    flux = np.zeros_like(DL_brute)
    for deg in range(pts_flux):
        flux += table_flux[deg] * DL_brute**(pts_flux - (deg+1))
    return flux

def DL_to_NUC(DL_brute):
    DL_brute[DL_brute > 2**14] = 2**14
    DL_corrige = np.zeros_like(DL_brute)
    for i in range(pts_NUC):
        DL_corrige += table_NUC[:,:,i] * DL_brute**(pts_NUC - (i+1))
    return DL_corrige


# Temps d'acquisition
metadata_CRED=pd.read_csv(f'./{work_path}/CRED/metadata.csv', encoding='unicode_escape', sep=',')
tim_python_CRED=np.asarray(metadata_CRED['t(s)'])
id_im_CRED=np.asarray(metadata_CRED['ImageUniqueID'])

# Chargement image
def load_CRED(id_):
    t_, id_ =tim_python_CRED[id_], id_im_CRED[id_]
    arr_raw=np.load(f'./{work_path}/CRED/{id_:06d}_{t_:.3f}.npy')
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
metadata_NIR=pd.read_csv(f'./{work_path}/XIQ/metadata.csv', encoding='unicode_escape', sep=',')
tim_python_NIR=np.asarray(metadata_NIR['t(s)'])
tim_XIMEA_NIR=np.asarray(metadata_NIR['XimeaSec']) + np.asarray(metadata_NIR['XimeaUSec'])*1e-6 
id_im_NIR=np.asarray(metadata_NIR['ImageUniqueID'])

# Chargement image
def load_NIR(id_):
    """ Entre 0 et 324 - 6423 pour MUR1 (essai recalage sur mur)  """
    t_, id_ =tim_python_NIR[id_], id_im_NIR[id_]
    img_NIR=np.zeros((channel, sub_width, sub_length))
    path=f"./{work_path}/XIQ/{id_:06d}_{t_:.3f}.npy"
    full_frame=np.load(path)[:-3,:-3]

    # Rangement des canneaux
    for cpt in range(25):
        i=int(cpt//np.sqrt(25))
        j=int(cpt%np.sqrt(25))
        img_NIR[cpt,:,:]=full_frame[i::5,j::5]

    return img_NIR


# --------------------------
# Identification des salves: 
# --------------------------
threshold = 5  # sec  ==> Changement de cordon pour identification

# CRED
time_diffs = np.diff(tim_python_CRED)
group_indices = np.where(time_diffs > threshold)[0] + 1
group_indices = np.concatenate(([0], group_indices, [len(tim_python_CRED)]))

info_CRED = []
for i in range(len(group_indices) - 1):
    start_idx = group_indices[i]
    end_idx = group_indices[i + 1] - 1
    group_times = tim_python_CRED[start_idx:end_idx + 1]
    info_CRED.append({
        'cordon': i + 1,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'times': group_times,
        'mean period':np.mean(np.diff(group_times)),
        'duration':group_times[-1]-group_times[0],
        'nb frame':len(group_times)
    })

# XIMEA
time_diffs = np.diff(tim_python_NIR)
group_indices = np.where(time_diffs > threshold)[0] + 1
group_indices = np.concatenate(([0], group_indices, [len(tim_python_NIR)]))

info_XIMEA = []
for i in range(len(group_indices) - 1):
    start_idx = group_indices[i]
    end_idx = group_indices[i + 1] - 1
    group_times = tim_python_NIR[start_idx:end_idx + 1]
    info_XIMEA.append({
        'cordon': i + 1,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'times': group_times,
        'mean period':np.mean(np.diff(group_times)),
        'duration':group_times[-1]-group_times[0],
        'nb frame':len(group_times)
    })



# -----------------------
# Travail sur un cordon : 
# -----------------------
CORDON=1
PERC=0.6
data_CRED=info_CRED[CORDON-1]
data_XIMEA=info_XIMEA[CORDON-1]

ref_time=min(data_CRED['times'][0], data_XIMEA['times'][0])
local_time_CRED=data_CRED['times']-ref_time
local_time_XIMEA=data_XIMEA['times']-ref_time

# Frequence plus faible on traite d'abord la XIMEA
relative_id_XIMEA=np.abs(local_time_XIMEA - PERC*data_XIMEA['duration']).argmin()
relative_time_XIMEA=local_time_XIMEA[relative_id_XIMEA]
absolute_id_XIMEA=relative_id_XIMEA+data_XIMEA['start_idx']

# Ensuite on prend l'image la plus proche de la CRED
relative_id_CRED=np.abs(local_time_CRED - relative_time_XIMEA).argmin()
relative_time_CRED=local_time_CRED[relative_id_CRED]
absolute_id_CRED=relative_id_CRED + data_CRED['start_idx']



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
im_XiQ_NIR=load_NIR(absolute_id_XIMEA)
im_CRED=load_CRED(absolute_id_CRED)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
cmp = 'viridis'

im1 = axes[0].imshow(im_CRED, cmap=cmp, vmin=0, vmax=2**14)
axes[0].set_title('CRED')
fig.colorbar(im1, ax=axes[0], orientation="horizontal")

im2 = axes[1].imshow(im_XiQ_NIR[8], cmap=cmp, vmin=0, vmax=255)
axes[1].set_title(f'XiQ NIR')
fig.colorbar(im2, ax=axes[1], orientation="horizontal")

plt.tight_layout()
plt.show()


#-------------------
# RECALAGE SPATIAL :
#-------------------

# Calcule de la matrice de transformation homographique
def get_H(img_C, img_XIQ, canal, see_match=False, see_result=False, savefig=False):
    img_ref=cv2.normalize(img_C, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_disto=cv2.normalize(img_XIQ, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sift = cv2.SIFT_create()
    # Détecter les points d'intérêt et calculer les descripteurs
    keypoints1, descriptors1 = sift.detectAndCompute(img_disto, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_ref, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Trouver les correspondances entre les descripteurs
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Appliquer le ratio test de Lowe pour filtrer les bonnes correspondances
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    if see_match:
        # Dessiner les correspondances
        result_image = cv2.drawMatches(img_disto, keypoints1, img_ref, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(result_image, cmap='gray')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        plt.grid(False)
        plt.tight_layout()
        if savefig:
            plt.savefig(f"./{work_path}/match_transformation/match_ch{canal}.png")
            plt.close(fig)
        else:
            plt.show()


    # Extraire les points d'intérêt correspondants
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Calculer l'homographie
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Aligner l'image 1 sur l'image 2
    height, width = WIDTH, DEPTH
    aligned_image = cv2.warpPerspective(img_disto, H, (width, height))

    if see_result:
        # Afficher les images
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        image=[img_disto, aligned_image]
        for i, ax in enumerate(axes):
            ax.imshow(image[i], cmap='viridis', vmin=0, vmax=255)
            ax.tick_params(labelsize=12) 
            # ax.axes.get_xaxis().set_ticks([])
            # ax.axes.get_yaxis().set_ticks([])
        plt.grid(False)
        plt.tight_layout()
        if savefig:
            plt.savefig(f"./{work_path}/transformation/match_ch{canal}.png")
            plt.close(fig)
        else:
            plt.show()


    return H


chanel = 19
H = get_H(img_C=im_CRED, img_XIQ=im_XiQ_NIR[chanel], canal=chanel+1,
        see_match=True, see_result=True, savefig=False)

if False:
    # Application de la matrice de transformation homographique
    matrice_H={}
    for ch in range(24):
        sub_im_XiQ_NIR=im_XiQ_NIR[ch]
        H = get_H(img_C=im_CRED, img_XIQ=sub_im_XiQ_NIR, canal=ch+1,
                see_match=True, see_result=True, savefig=True)
        matrice_H[ch]=H

    with open('H_matrix_mur.pkl', 'wb') as f:
        pickle.dump(matrice_H, f)


# Chargement des matrices de transformation
with open('./MIRE/H_matrix.pkl', 'rb') as f:
    matrice_H = pickle.load(f)


chanel=8
XIQ_raw=im_XiQ_NIR[chanel]
H_i=matrice_H[chanel]
XIQ_rescaled=cv2.warpPerspective(XIQ_raw, H_i, (DEPTH, WIDTH))
im_CRED_bin = im_CRED > 14000
aligned_image_bin = XIQ_rescaled > 100
mask_no_data=XIQ_rescaled==0

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axes[0].axes.get_xaxis().set_ticks([])
axes[0].axes.get_yaxis().set_ticks([])
axes[0].grid(False)
axes[0].imshow(im_CRED_bin, cmap='gray')

axes[1].axes.get_xaxis().set_ticks([])
axes[1].axes.get_yaxis().set_ticks([])
axes[1].grid(False)
plt.imshow(aligned_image_bin, cmap='gray')
plt.show()

import matplotlib.colors as mcolors
diff_arr=im_CRED_bin.astype(float)-aligned_image_bin.astype(float)
diff_arr[mask_no_data]=np.nan
fig, ax = plt.subplots()

bounds=[-1.5, -0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, ncolors=len(bounds)-1)
cmap = plt.get_cmap('viridis', len(bounds)-1)
cax=ax.imshow(diff_arr, cmap=cmap, norm=norm)
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
plt.show()
