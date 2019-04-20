import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage.color as color
import torch

def afficherImageLAB(img_lab):
    plt.imshow(color.lab2rgb(img_lab))
    plt.show()

def afficherImageRGB(img_rgb):
    plt.imshow(img_rgb)
    plt.show()

def afficherImageGrise(img_gray):
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
def show_images(images, titles):
    f = plt.figure()
    for i in range(1, len(images) + 1):
        subplot = f.add_subplot(1,len(images), i)
        # gray
        if (len(images[i-1].shape) == 2):
            plt.imshow(images[i-1], cmap='gray')
        else:
            plt.imshow(images[i-1])
        subplot.set_title(titles[i-1])

    plt.show(block=True)


def afficherPrediction(originale, grayscale, prediction):
    f = plt.figure()
    subplot = f.add_subplot(1,3, 1)
    plt.imshow(originale)
    subplot.set_title("originale")

    subplot = f.add_subplot(1,3, 2)
    plt.imshow(grayscale.reshape((256,256)), cmap='gray', vmin=0, vmax=1)
    subplot.set_title("gris")
    grayscale = grayscale.reshape((256,256,1))
    img_lab = color.rgb2lab(originale)
    test = img_lab[:, :, 0]
    test = test.reshape((256, 256, 1))
    lab = np.concatenate((test, prediction), axis=2)
    subplot = f.add_subplot(1,3, 3)
    print('lab2rgb')
    test = color.lab2rgb(lab)
    print('lab2rgb2')

    plt.imshow(color.lab2rgb(lab))
    print('lab2rgb3')

    subplot.set_title("prediction")

    plt.show()


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  plt.clf() # clear matplotlib 
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None: 
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))