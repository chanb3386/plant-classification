import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, regularizers
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
    species = [name for name in os.listdir("../flwrtrain") if name!="rys_plant_list.csv"]

    # https://www.tensorflow.org/tutorials/images/classification
    # normalizes data to 150x150 pixels
    image_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range = 90,  # random rotation
                                        brightness_range=[0.4,0.8]  # random brightness augmentation
                                        )
    train_data = image_generator.flow_from_directory(batch_size=17676,
                                         directory="../flwrtrain",
                                         shuffle=True,
                                         target_size=(128, 128),
                                         class_mode='binary')

    # images an array of plant images in array form
    train_images, train_labels = next(train_data)

    test_data = image_generator.flow_from_directory(batch_size=1060,
                                         directory="../flwrtest",
                                         shuffle=True,
                                         target_size=(128, 128),
                                         class_mode='binary')

    # images an array of plant images in array form
    test_images, test_labels = next(test_data)


    train_images = tf.image.rgb_to_grayscale(train_images)
    test_images = tf.image.rgb_to_grayscale(test_images)

    return np.array(train_images), np.array(test_images), np.array(train_labels), np.array(test_labels)



def createModel(d = 0):
    train_images, test_images, train_labels, test_labels = fetch_data()
    print(train_images.shape)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128,1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(80, activation='relu')) #kernel_regularizer=regularizers.l1(0.001),
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(20, activation="softmax"))

    ############################# plagiarized a wee bit
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=4, batch_size=32, validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.03, 0.2])
    plt.legend(loc='lower right')
    plt.show()

    model.evaluate(test_images,  test_labels, verbose=2)


    # SAVING MODEL
    # Can potentially add capability to save multiple models (if we want)
    model.save("test_model5.h5")
