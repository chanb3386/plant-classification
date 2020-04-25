import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage as sk
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# TODO: complete anymore data augmentation
# split data file into training and testing directories

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def createModel():
    # LOADING IMAGES

    # Load each directory of species (labels)
    species = [name for name in os.listdir("../data") if name!="rys_plant_list.csv"]

    # https://www.tensorflow.org/tutorials/images/classification
    # normalizes data to 150x150 pixels
    image_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range = 90,  # random rotation
                                        brightness_range=[0.2,1.0]  # random brightness augmentation
                                        )
    data = image_generator.flow_from_directory(batch_size=32,
                                         directory="../data",
                                         shuffle=True,
                                         target_size=(150,150),
                                         class_mode='binary')

    # images an array of plant images in array form
    images, _ = next(data)

    # data augmentation
    #for i in range(len(images)):
    #    images[i] = images[i].reshape(150,150,3)  # random rotation

    plotImages(images)




createModel()
