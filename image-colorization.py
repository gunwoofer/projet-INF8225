import numpy as np
import cv2
import skimage.color as color
import matplotlib.pyplot as plt
from colorizationNetwork import ColorizationNetwork
from image_utils import *
from dataset_utils import *
from google_images_download import google_images_download

def train(model, X, Y):
    pass


def main():
    # Image de test
    # X : Image en niveau de gris
    # Y : composantes a et b de l'image LAB
    (originale, Xtest, Ytest) = ouvrirImage("test/imagetest.png", affichage=False)

    # Telechargement Dataset
    if not os.path.exists("dataset"):
        telechargerDataSet(keywords="landscape", taille=100)
    (X, Y) = chargerDataset()

    # Creation du CNN
    model = ColorizationNetwork()

    # Entrainement du modele
    train(model, X, Y)


if __name__ == "__main__":
    main()
