import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage as sk
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# TODO: ADD MORE DATA AUGMENBTATION in fetch_data()

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def fetch_data():
    # LOADING IMAGES

    # Load each directory of species (labels)
    species = [name for name in os.listdir("../data") if name!="rys_plant_list.csv"]

    # https://www.tensorflow.org/tutorials/images/classification
    # normalizes data to 150x150 pixels
    image_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range = 90,  # random rotation
                                        brightness_range=[0.4,0.8]  # random brightness augmentation
                                        )
    data = image_generator.flow_from_directory(batch_size=5481,
                                         directory="../data",
                                         shuffle=True,
                                         target_size=(150,150),
                                         class_mode='binary')

    # images an array of plant images in array form
    images, labels = next(data)

    # plotImages(images)

    # Data augmentation on training set
    for i in range(len(images)):
        if i%4 == 0:
            images[i] = tf.image.flip_left_right(images[i])
        elif i%4 == 1:
            images[i] = tf.image.adjust_saturation(images[i],2)
        elif i%4 == 2:
            images[i] = tf.image.adjust_saturation(images[i],3)


    train_images, test_images = train_test_split(images, test_size = 0.1, random_state = 42)
    train_labels, test_labels = train_test_split(labels, test_size = 0.1, random_state = 42)

    train_images = tf.image.rgb_to_grayscale(train_images)
    test_images = tf.image.rgb_to_grayscale(test_images)

    return np.array(train_images), np.array(test_images), np.array(train_labels), np.array(test_labels)
    #return train_images, test_images, train_labels, test_labels


def createModel(d = 0):
    train_images, test_images, train_labels, test_labels = fetch_data()
    #for i in range (len(train_images)):
        #train_images[i] = train_images[i][0]
        #print(train_images[i], train_labels[i])
    print(train_images.shape)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(16, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(20, activation="softmax"))

    ############################# plagiarized a wee bit
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1)

    model.evaluate(test_images,  test_labels, verbose=2)

    # SAVING MODEL
    # Can potentially add capability to save multiple models (if we want)
    model.save("test_model.h5")
