# Importing all necessary libraries and frameworks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout,GlobalMaxPooling2D
from keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from keras.preprocessing import image

# Reading the excel file containing name of the image along with its label
df = pd.read_excel(r"/content/LabeledText.xlsx")

# Importing a Densenet 169 model 
pretrained_model2= tf.keras.applications.DenseNet169(
                   input_shape=(224,224,3),
                   weights='imagenet')

# Freezing all the layers
for layer in pretrained_model2.layers:
    layer.trainable = False

# Creating a model for generating Embeddings
embedding_model2 = Model(inputs=pretrained_model2.input, outputs=pretrained_model2.layers[-2].output)

# Defining a function which receives path to an image, loads and preprocesses it and returns it embedding
def get_embedding(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    embeddings = embedding_model2.predict(x)
    return embeddings

embeddings_list = []

# Iterate through all the records and generate their embeddings and adding it to the list
def generateImageEmbeddings():
    for index, row in df.iterrows():
        image_name = row['File Name']
        label_name=row['LABEL']
        image_name=image_name.replace(".txt",".jpg")
        folder_path = "/content/drive/MyDrive/Images/"
    
        image_path=folder_path+label_name+'/'+image_name
    
    
        embedding=get_embedding(image_path)
        embeddings_list.append(embedding)

    # Saving our numpy array containing all 4869 embeddings of size (1,1664)
    image_embeddings=np.array(embeddings_list)

    np.save("/content/densenet.npy",image_embeddings)
if __name__ == '__main__':
    generateImageEmbeddings()