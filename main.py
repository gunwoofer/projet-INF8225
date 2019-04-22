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
EPOCH = 10

def training(model, X, Y, epoch):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    i = 0
    nbtotal = X.shape[0]
    loss = 0
    for x, y in zip(X, Y):
        print("Epoch {} : Entrainement image {} sur {}".format(epoch, i, nbtotal))
        x = x.reshape((1, 1, 256, 256))
        y = y.reshape((1, 2, 256, 256))
        x, y = Variable(torch.from_numpy(x).float(), volatile=True).cuda(), Variable(torch.from_numpy(y).float()).cuda()
        yPredict = model(x)

        loss = criterion(yPredict, y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i = i + 1
    return loss

# Initialise selon l'article de recherche Richard Zhang, Phillip Isola, Alexei A. Efros. Colorful Image Colorization. University of California, Berkeley, 2016, disponible ici : https://arxiv.org/pdf/1603.08511.pdf
def init_google_net():
    net = cv2.dnn.readNetFromCaffe("./models/google_net_colorize.prototxt", "./models/google_net_colorize.caffemodel")
    pts = np.load("./models/google_net_colorize.npy")
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def main():
    # Image de test
    # X : Image en niveau de gris
    # Y : composantes a et b de l'image LAB
    # (originale, Xtest, Ytest) = ouvrirImage("test/street-test.jpg", affichage=False)
    (originale, Xtest, Ytest) = ouvrirImage("test/flower_test.jpg", affichage=False)
    # afficherPrediction(originale, Xtest, Ytest)
    # Telechargement Dataset
    # if not os.path.exists("dataset"):
    #     telechargerDataSet(keywords="landscape", taille=100)

    model = ColorizationNetworkv2()

    # if os.path.isfile('models/model-epoch-4-losses-0.003.pth'):
    #     model.load_state_dict(torch.load('models/model-epoch-4-losses-0.003.pth'))
    # else:

    (X, Y) = chargerDataset()
    # Creation du CNN

    # Recuperer un modele existant si besoin
    # model.load_state_dict(torch.load("models/model_benjamin.pth"))

    #Deployer le modele sur le gpu
    model.cuda()

    # Sinon entrainer un nouveau modele 
    losses = []
    for epoch in range(EPOCH):
        losses.append(training(model, X, Y, epoch))
    print("entrainement fini")

    plt.plot(losses, label="loss")
    plt.legend()
    plt.show()

    #Sauvegarde du modèle
    torch.save(model.state_dict(), 'models/model_benjamin.pth')
    print("sauvegarde du modele")

    # test
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

# Selon l'article de recherche Richard Zhang, Phillip Isola, Alexei A. Efros. Colorful Image Colorization. University of California, Berkeley, 2016, disponible ici : https://arxiv.org/pdf/1603.08511.pdf
def get_frame_googlenet(input):
    # init the google net model
    net = init_google_net()

    # Normalisation des données 
    scaled = input.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    #prediction
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # formatage
    ab = cv2.resize(ab, (input.shape[1], input.shape[0]))
    L = cv2.split(lab)[0]


    # Colorize image
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")

def get_frame_model(input, model_path):
    w,h = input.shape[0], input.shape[1]
    x = input
    model = ColorizationNetworkv2()

    if not os.path.isfile(model_path):
        print('error fichier n existe pas')
        return 
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    x = x.reshape((1, 1, w, h))
    x = x / 255
    # On passe la variable à cuda
    x_gpu =  Variable(torch.from_numpy(x).float(), volatile=True).cuda()
    # On calcul l'output
    output = model(x_gpu).cpu().detach().numpy().reshape((w, h, 2)) # output sous la forme ab

    #On prend les variables sous forme de tensor au bon format
    var1 = Variable(torch.from_numpy(x_gpu.cpu().detach().numpy().reshape(1, w, h)), volatile=True)
    var2 = Variable(torch.from_numpy(output.reshape(2, w, h)))

    # On converti le lab en rgb
    rgb_output = to_rgb(var1, var2)
    rgb_output *= 255
    return rgb_output.astype('uint8')

def get_frame(input, model="1"):
    if model == '1':
        #Architecture 2
        return get_frame_googlenet(input)
    else:
        #Architecture 1
        return get_frame_model(input, model)

if __name__ == "__main__": 
    main()
