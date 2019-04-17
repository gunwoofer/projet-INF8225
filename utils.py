import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage.color as color

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
            plt.imshow(images[i-1], cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(images[i-1])
        subplot.set_title(titles[i-1])

    plt.show(block=True)