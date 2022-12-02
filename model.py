from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten
from keras.utils import load_img, img_to_array
import keras.utils as image
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
classes = {
0:"Apple Bad", 
1:"Apple Good",   
2:"Apple mixed",  
3:"Banana Bad",   
4:"Banana Good",  
5:"Banana mixed", 
6:"Guava Bad",    
7:"Guava Good",   
8:"Guava mixed",  
9:"Lemon mixed",  
10:"Lime Bad",     
11:"Lime Good",    
12:"Orange Bad",   
13:"Orange Good",  
14:"Orange mixed",
15:"Pomegranate Bad",
16:"Pomegranate Good",
17:"Pomegranate mixed"
}
# def create_model():
#     model = Sequential()
#     model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(224,224,3)))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(32, (3,3), 1, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(64, (3,3), 1, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(128, (3,3), 1, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(256, (3,3), 1, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(512, (3,3), 1, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(18, activation='softmax'))
#     model.compile(optimizer = 'adam', loss="categorical_crossentropy", metrics=['accuracy'])
#     model.load_weights('model.h5')
#     return model

def create_model():
    model = tf.keras.models.Sequential([
    # first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(224,224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # third convolution layer
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # fourth convolution layer
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # flatten before feeding into Dense neural network. 
    tf.keras.layers.Flatten(),
    # 512 neurons in the hidden layer
    tf.keras.layers.Dense(512, activation="relu"),
    # 15 = 15 different categories
    # softmas takes a set of values and effectively picks the biggest one. for example if the output layer has
    # [0.1,0.1,0.5,0.2,0.1], it will take it and turn it into [0,0,1,0,0]
    tf.keras.layers.Dense(18, activation="softmax")
    ]);
    model.load_weights('model_3.h5')
    return model


def predict(file):
    model =create_model()
    img_pred = tf.keras.preprocessing.image.load_img('image.jpg',target_size = (224,224,3))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred,axis=0)
    reslt = model.predict(img_pred)
    return classes[np.argmax(reslt)]