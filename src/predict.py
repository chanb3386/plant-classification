import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage as sk
from skimage import io
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from matplotlib import image as im
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# @model: model saved in src folder, created in flower_class.py
# @image: name of the image in predict_images, ex: test.jpg
def predictNetwork(model,image):
    # opening image
    img = im.imread("predict_images/"+image)
    data = asarray(img)
    data = tf.image.resize(data, [150,150])
    data = tf.image.rgb_to_grayscale(data)

    predict = model.predict([[data]])

    species = [name for name in os.listdir("../data") if name!="rys_plant_list.csv"]
    index = predict[0].argmax()
    print(species[index])
# TODO display images
