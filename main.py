import numpy as np
import torch
import cv2
import torch.nn as nn
import skimage.color as color
import matplotlib.pyplot as plt
from colorizationNetwork import ColorizationNetwork
from image_utils import *
from dataset_utils import *
from google_images_download import google_images_download

BATCH_SIZE = 10
LEARNING_RATE = 0.01


def train(model, X, Y):
    print("Debut de l'entrainement..")
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    i = 0
    nbtotal = X.shape[0]
    for x, y in zip(X, Y):
        print("Entrainement sur l'image {} sur {}".format(i, nbtotal))
        x = x.reshape((1, 1, 256, 256))
        y = y.reshape((1, 2, 256, 256))
        yPredict = model(torch.from_numpy(x).float())

        loss = criterion(yPredict, torch.from_numpy(y).float())
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i = i + 1
    print("Entrainement de l'epoch terminé !")



def main():
    # Image de test
    # X : Image en niveau de gris
    # Y : composantes a et b de l'image LAB
    (originale, Xtest, Ytest) = ouvrirImage("test/landscapetest.jpg", affichage=False)

    # Telechargement Dataset
    if not os.path.exists("dataset"):
        telechargerDataSet(keywords="landscape", taille=100)
    (X, Y) = chargerDataset()

    # Creation du CNN
    model = ColorizationNetwork()

    # Entrainement du modele
    train(model, X, Y)

    # Petit test rapide pour rigoler
    Xtest = Xtest.reshape((1, 1, 256, 256))
    output = model(torch.from_numpy(Xtest).float()).detach().numpy()
    
    output = output.reshape((256,256,2))
    output = output * 128
    Xtest = Xtest.reshape((256, 256, 1))
    save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
    save_name = 'img-test.jpg'
    to_rgb(Xtest, output, save_path=save_path, save_name=save_name)




if __name__ == "__main__":
    main()
