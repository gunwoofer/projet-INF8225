import numpy as np
import torch
import cv2
import torch.nn as nn
import skimage.color as color
import matplotlib.pyplot as plt
from colorizationNetwork import ColorizationNetworkv2
from image_utils import *
from dataset_utils import *
from torch.autograd import Variable


BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 1

def training(model, X, Y, epoch):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    losses = []
    i = 0
    nbtotal = X.shape[0]
    for x, y in zip(X, Y):
        print("Epoch {} : Entrainement image {} sur {}".format(epoch, i, nbtotal))
        x = x.reshape((1, 1, 256, 256))
        y = y.reshape((1, 2, 256, 256))
        x, y = Variable(torch.from_numpy(x).float(), volatile=True).cuda(), Variable(torch.from_numpy(y).float()).cuda()
        yPredict = model(x)

        loss = criterion(yPredict, y)
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
    # (originale, Xtest, Ytest) = ouvrirImage("test/street-test.jpg", affichage=False)
    (originale, Xtest, Ytest) = ouvrirImage("test/flower_test.jpg", affichage=False)

    # Telechargement Dataset
    # if not os.path.exists("dataset"):
    #     telechargerDataSet(keywords="landscape", taille=100)
    (X, Y) = chargerDataset()

    # Creation du CNN
    model = ColorizationNetworkv2()

    # Recuperer un modele existant si besoin
    # model.load_state_dict(torch.load("models/model_benjamin.pth"))

    #Deployer le modele sur le gpu
    model.cuda()

    # Sinon entrainer un nouveau modele 
    for epoch in range(EPOCH):
        training(model, X, Y, epoch)
    print("entrainement fini")

    #Sauvegarde du modèle
    torch.save(model.state_dict(), 'models/model_benjamin.pth')
    print("sauvegarde du modele")

    # Petit test rapide pour rigoler
    Xtest = Xtest.reshape((1, 1, 256, 256))
    Xtest_gpu =  Variable(torch.from_numpy(Xtest).float(), volatile=True).cuda()
    output = model(Xtest_gpu).cpu().detach().numpy()
    
    print("output genere")
    output = output.reshape((256,256,2))
    output = output * 128
    print("output traite")
    Xtest = Xtest.reshape((256, 256, 1))
    
    print("affichage")
    afficherPrediction(originale, Xtest, output)




if __name__ == "__main__":
    main()
