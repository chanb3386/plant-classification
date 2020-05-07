import flower_class as fc
import predict
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

if __name__ == "__main__":
    while(True):
        a = input("Enter an option: [TRAIN, PREDICT, VIEWLOG, END]: ")
        check = a.lower()

        if check == "train":
            fc.createModel()
        elif check == "predict":
            print("Loading model")
            model = keras.models.load_model("test_model.h5")
            name = input("Enter image name: ")
            predict.predictNetwork(model,name)
        elif check == "viewlog":
            # TODO
            # do prediction over entire test set, print out to a txt doc
            print("viewlog")
        elif check == "end":
            break;
        else:
            continue
