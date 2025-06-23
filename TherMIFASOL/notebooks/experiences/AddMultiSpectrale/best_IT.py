""" Lecture des données capteurs."""
""" Code adapté pour la CRED mes developpements """

# Imports
import os
import cv2
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from datetime import datetime
import time
from skimage.measure import label,regionprops
from scipy import ndimage as ndi

plt.style.use(['science', 'notebook', 'grid'])
np.seterr(divide='ignore', invalid='ignore')

work_path='recherche_IT'
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
Tsol = 1554  # K
Tliq = 1625  # K
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
def get_metadata_CRED(IT):
    tim_python_NIR = np.loadtxt(f'./{work_path}/CRED/CRED_{work_path}_{IT}/time.txt')
    return tim_python_NIR

# Chargement image
def load_CRED(IT, id_):
    arr_raw=np.load(f'./{work_path}/CRED/CRED_{work_path}_{IT}/data/im_{id_}.npy')
    metadata=np.zeros_like(arr_raw, dtype=bool)
    metadata[0,:4]=True
    arr = np.ma.masked_array(arr_raw, mask=metadata)
    return arr

#========================================================
#                     Caméra XIMEA
#========================================================
# Une différence de 2 heures (UTC+2) liés aux horloges internes 
# des deux programmes ne rendent plus coherents les date 
# d'enregistrement
Deux_heures = 2*60*60  # 2 heures en sec
# Metadata
# im_XiQ_NIR : 'float64'
channel=25
sub_width=int((1088-3)//np.sqrt(channel))
sub_length=int((2048-3)//np.sqrt(channel))

def get_metadata_XIQ(IT):
    metadata_NIR=pd.read_csv(f'./{work_path}/XIQ/XIQ_{work_path}_{IT}/metadata.csv', encoding='unicode_escape', sep=',')
    tim_python_NIR=np.asarray(metadata_NIR['t(s)'])
    exposure=np.asarray(metadata_NIR['ExposureTime']).mean()
    id_im_NIR = np.asarray(metadata_NIR['ImageUniqueID'])

    absolute_time = np.asarray([
        time.mktime(datetime.strptime(t, "%Y:%m:%d %H:%M:%S").timetuple()) \
             for t in metadata_NIR['DateTimeOriginal']
    ]) + np.asarray(metadata_NIR['SubsecTimeOriginal']) + Deux_heures

    return tim_python_NIR, id_im_NIR, absolute_time, exposure

# Chargement image
def load_NIR(IT, t_, id_):
    """ Entre 0 et 324 - 6423 pour MUR1 (essai recalage sur mur)  """
    img_NIR=np.zeros((channel, sub_width, sub_length))
    path=f"./{work_path}/XIQ/XIQ_{work_path}_{IT}/{id_:06d}_{t_:.3f}.npy"
    full_frame=np.load(path)[:-3,:-3]

    # Rangement des canneaux
    for cpt in range(25):
        i=int(cpt//np.sqrt(25))
        j=int(cpt%np.sqrt(25))
        img_NIR[cpt,:,:]=full_frame[i::5,j::5]

    return img_NIR

# Chargement des matrices de transformation
with open('./MIRE/H_matrix.pkl', 'rb') as f:
    matrice_H = pickle.load(f)



#========================================================
#               GESTION TEMPS : SYNCHRO
#========================================================
def get_id_syncrhonize(IT, perc_pick_img, plot_=False):

    # Get time recording
    time_CRED=get_metadata_CRED(IT)
    time_XIQ, ID_im_XIQ, abs_time_XIQ, _ =get_metadata_XIQ(IT)
    print(f"CRED: {datetime.fromtimestamp(time_CRED[0])}")
    print(f"XIMEA: {datetime.fromtimestamp(abs_time_XIQ[0])}")

    start_same_record=max(time_CRED[0], abs_time_XIQ[0])
    stop_same_record=min(time_CRED[-1], abs_time_XIQ[-1])
    duration_same_recording=stop_same_record-start_same_record

    # off_set=min(time_CRED[0], abs_time_XIQ[0])
    # plt.plot(time_CRED-off_set, np.arange(len(time_CRED)), label='CRED')
    # plt.plot(abs_time_XIQ-off_set, np.arange(len(abs_time_XIQ)), label='XIQ')
    # plt.legend()
    # plt.show()

    # Frequence plus faible on traite d'abord la XIMEA
    relative_id_XIMEA=np.abs(
        abs_time_XIQ - (start_same_record + perc_pick_img*duration_same_recording)
        ).argmin()
    picked_time_XIMEA=abs_time_XIQ[relative_id_XIMEA]

    # Ensuite on prend l'image la plus proche de la CRED
    relative_id_CRED=np.abs(time_CRED - picked_time_XIMEA).argmin()
    picked_time_CRED=time_CRED[relative_id_CRED]

    if plot_:
        extr=0.12
        fig, ax = plt.subplots()

        ax.vlines(
            [picked_time_CRED-start_same_record, picked_time_XIMEA-start_same_record],
            ymin=-0.5,
            ymax=1.5,
            color=['b', 'g'],
            linestyles='dotted'
            )
        ax.scatter([0],[0], facecolor='white', edgecolor='black', label="Prise d'image")
        ax.plot([0,0],[0,0], c='black', label="Images selctionnées", linestyle='dotted')

        ax.scatter(time_CRED-start_same_record, np.ones(len(time_CRED))*0, facecolor='blue', edgecolor='black')
        ax.scatter(abs_time_XIQ-start_same_record, np.ones(len(abs_time_XIQ))*1, facecolor='green', edgecolor='black')

        ax.set_yticklabels(['', '', 'CRED','','','', 'NIR', ''])
        ax.set_ylim(-0.5, 1.5)
        mean_=(picked_time_XIMEA + picked_time_CRED)/2 -start_same_record
        ax.set_xlim(mean_-extr, mean_+extr)
        ax.legend()
        ax.set_xlabel('Temps[sec]')
        plt.title(f"""Prises d'image depuis déclanchement par trig""")
        plt.show()

    return relative_id_CRED, relative_id_XIMEA

def show_img_synchro(IT, perc, chanel):

    time_XIQ, ID_im_XIQ, _, exposure = get_metadata_XIQ(IT=IT)
    relative_id_CRED, relative_id_XIMEA = get_id_syncrhonize(
        IT=IT,
        perc_pick_img=perc
    )

    t_X, id_X = time_XIQ[relative_id_XIMEA], ID_im_XIQ[relative_id_XIMEA]
    im_XiQ_NIR=load_NIR(IT=IT, t_=t_X, id_=id_X)
    im_CRED=load_CRED(IT=IT, id_=relative_id_CRED)

    XIQ_raw=im_XiQ_NIR[chanel]
    H_i=matrice_H[chanel]
    XIQ_rescaled=cv2.warpPerspective(XIQ_raw, H_i, (DEPTH, WIDTH))

    # Comparaison images brutes
    fig, axes = plt.subplots(1, 2)
    cmp = 'viridis'
    im1 = axes[0].imshow(im_CRED, cmap=cmp, vmin=0, vmax=2**14)
    axes[0].set_title('CRED')
    fig.colorbar(im1, ax=axes[0], orientation="horizontal")
    im2 = axes[1].imshow(XIQ_rescaled, cmap=cmp, vmin=0, vmax=255)
    axes[1].set_title(f'XiQ NIR {chanel+1}')
    fig.colorbar(im2, ax=axes[1], orientation="horizontal")
    plt.tight_layout()
    plt.show()


#========================================================
#            REPERAGE D'UNE ZONE D'INTERET
#========================================================
def get_mask_ROI(image:np.array, T_inf:float, T_sup:float):

    mask_mushy = (image > T_inf) & (image < T_sup)
    mask_inf = (image > T_inf-5) & (image < T_inf+5)
    mask_sup = (image > T_sup-5) & (image < T_sup+5)

    # pos_=(np.argmax(mask_mushy==1)%640, np.argmax(mask_mushy==1)//640)
    # depth_concerned=512-np.argmax(mask_mushy==1)//640
    # area_detected_as_mushy=np.sum(mask_mushy)
    # tot_area=640*depth_concerned
    mushy = np.zeros_like(image)
    mushy_label = label(mask_mushy) 
    # Tri des régions par taille de surface
    rps = np.asarray(regionprops(mushy_label))
    areas = np.asarray([r.area for r in rps])  # Number of pixels of the area
    idxs = np.argsort(areas)[::-1]

    flag_ind=np.argmax(areas[idxs]/areas[idxs[0]] > 0.25)

    # Merge zones with an area higher than 25% compared to the biggest area detected.
    for idx in idxs[:flag_ind+1]:
        mushy[tuple(rps[idx].coords.T)] = 1

    # Remplir les trous dans les masques
    mushy = ndi.binary_fill_holes(mushy == 1)
    y_sup_thermo, x_sup_thermo = np.where(mushy&mask_sup)
    y_inf_thermo, x_inf_thermo = np.where(mushy&mask_inf)

    return mushy, (y_sup_thermo, x_sup_thermo), (y_inf_thermo, x_inf_thermo)

def save_ROI_img_synchro(img_therm:np.array, mushy:np.array, XIQ_rescaled:np.array, chanel:int, save_path:str):

    transparent_mushy=np.ones((mushy.shape[0], mushy.shape[1], 4), dtype=int)*230
    transparent_mushy[mushy,3]=0 
    color_inf='r'
    color_sup='r'

    fig, axes = plt.subplots(1, 2)

    im1 = axes[0].imshow(img_therm, interpolation='nearest', vmin=700, vmax=1800, cmap='viridis')
    axes[0].imshow(transparent_mushy)
    # axes[0].scatter(x_liq_thermo, y_liq_thermo, c=color_inf, s=0.1, label='Borne inf.')
    # axes[0].scatter(x_sol_thermo, y_sol_thermo, c=color_sup, s=0.1, label='Borne sup.')
    #ax.scatter(x_liq_thermo, y_liq_thermo, c='b', s=1, label='Liquidus')
    axes[0].set_title('CRED')

    cbar = fig.colorbar(im1, orientation='horizontal', pad=0.1, label="température [K]")
    for value, color in zip([T_inf, T_sup], [color_inf, color_sup]):
        cbar.ax.axvline(value, color=color, linestyle='-', linewidth=2)

    im2 = axes[1].imshow(XIQ_rescaled, cmap='viridis', vmin=0, vmax=255)
    axes[1].imshow(transparent_mushy)
    axes[1].set_title(f'XiQ NIR {chanel+1}')
    cbar2 = fig.colorbar(im2, orientation='horizontal', pad=0.1, label="Niveau de gris [ADU]")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)



#========================================================
#              RECHERCHE DU MEILLEUR IT
#========================================================
# Selection des deux canaux extremum
ch_max=4
ch_min=20

# Matrices de transformation homographique
H_max=matrice_H[ch_max]
H_min=matrice_H[ch_min]

# Définition des bornes de la zone d'intérêt
T_inf=Tsol-100
T_sup=Tsol+20

# Liste des identifiants de temps d'intégration
# '3_retour', '4_retour', '5_retour': en dehors du champs de vue de la caméra.
it_label=[
    '1', '2', '3', '4', '5', '6', '7', '8', '9', \
    '10', '11_bis', '12'
    ]
PERC=[
    0.8, 0.5, 0.8, 0.8, 0.7, 0.8, 0.6, 0.7, \
    0.7, 0.7, 0.8, 0.8
]

save_img=False
image_save_path=f"./{work_path}/resultats/ROI_histogramme"
histogram_save_path=f"./{work_path}/resultats/histogrammes"

my_bins=np.arange(0, 256, 5)
np.save(f"{histogram_save_path}/bins_all_chanel", my_bins)

for it, perc in zip(it_label, PERC):
    print(it, perc)

    # Recupération des métadata des deux caméras
    ## Temps absolu CRED
    time_CRED=get_metadata_CRED(IT=it)
    ## (Temps et ID crappy) + temps absolu XIMEA
    t_NIR, id_NIR, time_XIQ, exposure=get_metadata_XIQ(IT=it)
    print(f"IT{it} -- {exposure}")

    # Synchronization des données et récupération du meilleur candidat
    id_CRED, id_XIQ = get_id_syncrhonize(IT=it, perc_pick_img=perc, plot_=False)

    # Chargement des images associées
    im_CRED=load_CRED(IT=it, id_=id_CRED)
    im_XIQ=load_NIR(IT=it, t_=t_NIR[id_XIQ], id_=id_NIR[id_XIQ])
    
    # Recalage des images aux canaux extremes
    img_max_rescaled=cv2.warpPerspective(im_XIQ[ch_max], H_max, (DEPTH, WIDTH))
    img_min_rescaled=cv2.warpPerspective(im_XIQ[ch_min], H_min, (DEPTH, WIDTH))

    # Calcul de la carte thermique
    thermo_CRED=DL_to_thermo(im_CRED.astype(float), eps=0.32)

    # Repérage de la zone d'interet.
    mask_ROI, *reste = get_mask_ROI(thermo_CRED, T_inf, T_sup)
    if save_img:
        save_ROI_img_synchro(
            img_therm=thermo_CRED, 
            mushy=mask_ROI, 
            XIQ_rescaled=img_max_rescaled, 
            chanel=ch_max,
            save_path=f'{image_save_path}/{it}_max.png'
            )
        save_ROI_img_synchro(
            img_therm=thermo_CRED, 
            mushy=mask_ROI, 
            XIQ_rescaled=img_min_rescaled, 
            chanel=ch_min,
            save_path=f'{image_save_path}/{it}_min.png'
            )

    hist, bins = np.histogram(img_min_rescaled[mask_ROI], bins=my_bins)
    np.save(f"{histogram_save_path}/{it}_values_{ch_min}", hist)
    hist, bins = np.histogram(img_max_rescaled[mask_ROI], bins=my_bins)
    np.save(f"{histogram_save_path}/{it}_values_{ch_max}", hist)




#========================================================
#             AFFICHAGE DES HISTOGRAMMES
#========================================================
my_bins = np.load(os.path.join(histogram_save_path, 'bins_all_chanel.npy'))

# Initialize the figure and axes
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
axs = axs.flatten()

for i, it in enumerate(it_label):

    *_, it_value = get_metadata_XIQ(it) 
    # Load the histogram data for channel 4 and 20
    hist_4 = np.load(os.path.join(histogram_save_path, f"{it}_values_4.npy"))
    hist_20 = np.load(os.path.join(histogram_save_path, f"{it}_values_20.npy"))

    # Plot the histograms on the same graph
    axs[i].bar(my_bins[:-1], hist_4, width=np.diff(my_bins), alpha=0.5, label='Channel 4')
    axs[i].bar(my_bins[:-1], hist_20, width=np.diff(my_bins), alpha=0.5, label='Channel 20')
    
    # Set the title and labels
    axs[i].set_title(f"Integration Time {it_value}")
    axs[i].set_xlabel("Pixel Intensity")
    axs[i].set_ylabel("Frequency")
    axs[i].legend()

# Adjust the layout
plt.tight_layout()
plt.show()