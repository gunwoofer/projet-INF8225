import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from colorizationNetwork import ColorizationNetwork
import skimage.color as color
import skimage.io as io
import matplotlib.pyplot as plt
from utils import *

SIZE_IMAGE = 256


def ouvrirImage(affichage = False):
    # Ouvre l'image en RGB et rescale
    img_rgb = cv2.imread("dataset/imagetest.png")[...,::-1]
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


def main():
    # X : Image en niveau de gris
    # Y : composantes a et b de l'image LAB
    (originale, X, Y) = ouvrirImage(affichage=True)
    # Creation du CNN
    model = ColorizationNetwork()

if __name__ == "__main__":
    main()
