import numpy as np
import cv2
import skimage.color as color
import matplotlib.pyplot as plt
from colorizationNetwork import ColorizationNetwork
from image_utils import *
from dataset_utils import *
from google_images_download import google_images_download

def train(model):
    pass


def main():
    # Image de test
    # X : Image en niveau de gris
    # Y : composantes a et b de l'image LAB
    (originale, X, Y) = ouvrirImage("test/imagetest.png", affichage=False)

    # Telechargement Dataset
    telechargerDataSet(keywords="landscape", taille=100)

    # Creation du CNN
    model = ColorizationNetwork()

    # Entrainement du modele
    train(model)

    
if __name__ == "__main__":
    main()
