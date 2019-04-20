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
        print(loss)
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
    (originale, Xtest, Ytest) = ouvrirImage("test/bateau.jpg", affichage=False)
    # afficherPrediction(originale, Xtest, Ytest)
    # Telechargement Dataset
    # if not os.path.exists("dataset"):
    #     telechargerDataSet(keywords="landscape", taille=100)

    model = ColorizationNetworkv2()

    if os.path.isfile('models/model-epoch-4-losses-0.003.pth'):
        model.load_state_dict(torch.load('models/model-epoch-4-losses-0.003.pth'))
    else:

        (X, Y) = chargerDataset()
        # Creation du CNN

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
    model.cuda()
    output = model(Xtest_gpu).cpu().detach().numpy()
    
    # print("output genere")
    # output = output.reshape((256,256,2))
    # output *= 128
    # print("output traite")
    # Xtest = Xtest.reshape((256, 256, 1))
    
    # print("affichage")
    #afficherPrediction(originale, Xtest, output)
    var1 = Variable(torch.from_numpy(Xtest_gpu.cpu().detach().numpy().reshape(1, 256, 256)), volatile=True)
    var2 = Variable(torch.from_numpy(output.reshape(2, 256, 256)))

    to_rgb(var1, var2, afficher=True)

def get_frame(input):
    w,h = input.shape[0], input.shape[1]
    lab = color.rgb2lab(1.0/255*input)
    ab = lab[:, :, 1:]
    x = lab[:, :, 0]
    model = ColorizationNetworkv2()
    model.load_state_dict(torch.load('models/model-epoch-4-losses-0.003.pth'))
    model.cuda()

    x = x.reshape((1, 1, w, h))
    x_gpu =  Variable(torch.from_numpy(x).float(), volatile=True).cuda()
    output = model(x_gpu).cpu().detach().numpy().reshape((w, h, 2)) # output sous la forme ab
    x = x.reshape(( w,h, 1 ))
    new_lab = np.concatenate((x, output), axis=2)
    rgb_output = color.lab2rgb(new_lab)
    rgb_output *= 255
    plt.imshow(rgb_output.astype('uint8'))
    plt.show()
    return rgb_output.astype('uint8')
if __name__ == "__main__":
    main()
