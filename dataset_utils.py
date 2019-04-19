import numpy as np
import cv2
import skimage.color as color
import matplotlib.pyplot as plt
import os
from google_images_download import google_images_download

SIZE_IMAGE = 256

def telechargerDataSet(keywords, taille):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":keywords, "limit":taille, "print_urls":True, "format":"jpg", "no_directory":True, "output_directory":"dataset"}   
    paths = response.download(arguments)   
    print("Dataset chargé !")   
    i = 0
    for filename in os.listdir("dataset"): 
        dst ="image" + str(i) + ".jpg"
        src ="dataset/" + filename 
        dst ="dataset/" + dst 
        os.rename(src, dst) 
        i += 1

def chargerDataset():
    datasetX = []
    datasetY = []
    i = 0
    for image in os.listdir("dataset"):
        if (image.endswith('.jpg')):
            (_, X, Y) = ouvrirImage("dataset/" + image, affichage=False)
            datasetX.append(X)
            datasetY.append(Y/128)  
            print("image " + str(i) + "bien ajouté")
            if i == 99:
                print("fin de la recuperation des images")
                break
            i += 1
        # les composantes a et b sont entre -128 et 128
        # comme notre modele retourne un truc entre -1 et 1 (avec tanh a la fin)
        # il faut aussi diviser par 128 nos label pour bien calculer la perte

    return (np.array(datasetX), np.array(datasetY))

def ouvrirImage(path, affichage = False):
    '''
    Ouvre une image 
    img_rgb : Image originale
    img_gray : Image en niveau de gris
    colors : composantes a et b de l'image LAB
    Retourne (img_rgb, img_gray, colors)
    '''

    # Ouvre l'image en RGB et rescale
    img_rgb = cv2.imread(path)[...,::-1]
    img_rgb = cv2.resize(img_rgb, (SIZE_IMAGE,SIZE_IMAGE), interpolation = cv2.INTER_AREA)
    # Converti en LAB
    img_lab = color.rgb2lab(img_rgb)
    # Grayscale part
    img_gray = img_lab[:,:,0]
    # Color part
    colors = img_lab[:,:,1:] 
    # Affichage initiale
    if (affichage):
        show_images([img_rgb, img_gray], titles=["Originale", "Niveau de gris"])

    return (img_rgb, img_gray, colors)