
import os
import cv2
import numpy as np

#========================================================
#                OpenCV
#========================================================
def get_video_from_images(image_folder, output_video, fps):

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    height, width, layers = frame.shape
    size = (width, height)

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        out.write(img)

    out.release()
    print(f"Vidéo générée : {output_video}")

#========================================================
#                ImageMagick
#========================================================
def crop_image(input_image_path, output_image_path, crop_width, crop_height, crop_x, crop_y):
    """
    Cette fonction utilise ImageMagick pour cropper une image.
    
    :param input_image_path: Chemin vers l'image source
    :param output_image_path: Chemin vers l'image de sortie
    :param crop_width: Largeur du crop
    :param crop_height: Hauteur du crop
    :param crop_x: Position x du crop
    :param crop_y: Position y du crop
    """
    command = f"convert {input_image_path} -crop {crop_width}x{crop_height}+{crop_x}+{crop_y} {output_image_path}"
    os.system(command)

def batch_crop_images(filename, input_folder, output_folder, crop_width, crop_height, crop_x, crop_y):
    """
    Cette fonction batch crope toutes les images dans un dossier donné.
    
    :param input_folder: Dossier contenant les images source
    :param output_folder: Dossier où les images cropées seront sauvegardées
    :param crop_width: Largeur du crop
    :param crop_height: Hauteur du crop
    :param crop_x: Position x du crop
    :param crop_y: Position y du crop
    """

    input_image_path = os.path.join(input_folder, filename)
    output_image_path = os.path.join(output_folder, filename)
    #print(f"Image croppée: {input_image_path}")
    crop_image(input_image_path, output_image_path, crop_width, crop_height, crop_x, crop_y)
    #print(f"Image croppée: {output_image_path}")

def join_images(image1_path, image2_path, image3_path, output_image_path):
    """
    Cette fonction joint trois images en une seule avec deux colonnes.
    La première colonne contient deux images en deux lignes, et la deuxième colonne contient la troisième image.

    :param image1_path: Chemin vers la première image (première ligne de la première colonne)
    :param image2_path: Chemin vers la deuxième image (deuxième ligne de la première colonne)
    :param image3_path: Chemin vers la troisième image (deuxième colonne)
    :param output_image_path: Chemin de l'image de sortie
    """
    # Créer une image temporaire qui empile verticalement les deux premières images
    temp_image_path = "temp_vertical_stack.jpg"
    command_stack =  f'convert {image1_path} {image2_path} -append {temp_image_path}'
    os.system(command_stack)

    # Joindre l'image temporaire avec la troisième image côte à côte (horizontalement)
    command_join = f'convert {temp_image_path} {image3_path} +append {output_image_path}'
    os.system(command_join)

    # Supprimer l'image temporaire
    os.remove(temp_image_path)

    # print(f"Image finale créée: {output_image_path}")


# Example
if __name__ == "__name__":
    input_folder = "/home/dolle/Bureau/GradientThermique/EPS40_TEMPOR_LONGUE_bis"
    output_folder = "/home/dolle/Bureau/GradientThermique/EPS40_TEMPOR_LONGUE_bis/crop/"
    crop_width = 1005
    crop_height = 318
    crop_x = 104
    crop_y = 188


    for i in np.arange(1, 360):
        filename = f"R_TEMPOR_LONGUE_C25_{i}.png"
        # batch_crop_images(filename, input_folder, output_folder, crop_width, crop_height, crop_x, crop_y)
        folder_cropp = "/home/dolle/Bureau/GradientThermique/EPS40_TEMPOR_LONGUE_bis/crop/"
        folder_Hunt = "/home/dolle/Bureau/GradientThermique/EPS40_TEMPOR_LONGUE_bis/"

        image1_path = f'{folder_cropp}/GRAD_TEMPOR_LONGUE_C25_{i}.png'
        image2_path = f'{folder_cropp}/R_TEMPOR_LONGUE_C25_{i}.png'
        image3_path = f'{folder_Hunt}/Hunt_{i}.png'
        output_image_path = f'{folder_Hunt}/res/im_{i}.png'
        join_images(image1_path, image2_path, image3_path, output_image_path)

#========================================================
