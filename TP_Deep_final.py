import tensorflow as tf
from tensorflow import keras
import numpy as np
directory = 'dataset'
import matplotlib.pyplot as plt
import cv2
import os 
import pandas as pd 
import matplotlib.image as mpimg
import random


def create_dataset(img_folder):
   
    # to store all images  
    img_data_array = []
    class_name=[]
    
    # browse directories 
    for dir1 in os.listdir(img_folder):
        
        # browse files from that directory 
        for file in os.listdir(os.path.join(img_folder, dir1)):
            
            # join path 
            image_path = os.path.join(img_folder, dir1,  file)
            # read the image
            image = cv2.imread(image_path)
            # resize the image 
            image = cv2.resize(image, (270, 480),interpolation = cv2.INTER_AREA)
            # image as np array 
            image = np.array(image).reshape(270, 480, 3)
            # array type as float 
            image = image.astype('float32')
            # normalizing between 0 and 1
            image /= 255 
            # store image 
            img_data_array.append(image)
            # get class name by folder's name 
            class_name.append(int(dir1))
            
    return img_data_array, class_name

# create lists of dataset and their labels 
img_data, class_name =create_dataset('dataset')

# from list to np arrays 
movies_data = np.array(img_data, np.float32)
movies_class = np.array(class_name, np.int32).reshape(-1 ,1)

# fonction that assures two arrays are shuffled in the same manner 
def shuffle_in_unison(a, b):
    # make sure the vectors have the same length 
    assert len(a) == len(b)
    # random shuffle 
    p = np.random.permutation(len(a))
    # return shuffled array 
    return a[p], b[p]

# shuffle data [the movies images and their labels]
movies_data_shuffled, movies_class_shuffled = shuffle_in_unison(movies_data, movies_class)

# split data images by 2 [train and test] 
Xtrain, Xtest = np.split(movies_data_shuffled, 2)
# split data labels by 2 [train and test]
Ytrain, Ytest = np.split(movies_class_shuffled, 2)

# creating the model 
model = keras.models.Sequential()

# input layer as 270*480*3 [rgb channels]
model.add( keras.layers.Input((270,480,3)) )
#convolution layer with 8 filters (8 neurons) and 3 by 3 window size for the activation map 
model.add( keras.layers.Conv2D(8, (3,3),  activation='relu') )
# max pooling the 2 by 2 window size 
model.add( keras.layers.MaxPooling2D((2,2)))
# convolution layer with 16 filters (16 neurons) and 3 by 3 window size for the activation map 
model.add( keras.layers.Conv2D(16, (3,3), activation='relu') )
# max pooling 
model.add( keras.layers.MaxPooling2D((2,2)))
# flatten the data for the the fully connected layers 
model.add( keras.layers.Flatten())
# dense layer with 100 output 
model.add( keras.layers.Dense(100, activation='relu'))

# dense layer with 2 outputs for our classes 0 or 1 with softmax as activation function (percentage of the two classes)
model.add( keras.layers.Dense(2, activation='softmax'))
model.summary()

# compile model 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# batch_size 
batch_size  = 6
# number of epoch 
epochs      =  10

# store model history and fit the model to our data 
history = model.fit(  Xtrain, Ytrain,
                      batch_size      = batch_size,
                      epochs          = epochs,
                      verbose         = 1,
                      validation_data = (Xtest, Ytest))

# test our model 
score = model.evaluate(Xtest, Ytest, verbose=0)

print(f'Test loss     : {score[0]:4.4f}')
print(f'Test accuracy : {score[1]:4.4f}')

model.save("my_model")

np.save('model_history.npy', history.history)

history_plot = np.load('model_history.npy',allow_pickle='TRUE').item()

## show plot 

#%%

# show results 

plt.plot(np.arange(1, epochs + 1), np.array(history_plot['loss']).T, c = 'b', label = 'loss')
plt.plot(np.arange(1, epochs + 1), np.array(history_plot['val_loss']).T, '--', c = 'b', label = 'val_loss')
plt.legend(bbox_to_anchor=(1.5, 1), loc = 'upper right')

plt.show()

plt.plot(np.arange(1, epochs + 1), np.array(history_plot['accuracy']).T, c = 'r', label = 'accuracy')
plt.plot(np.arange(1, epochs + 1), np.array(history_plot['val_accuracy']).T,'--', c = 'r', label = 'val_accuracy')

plt.legend(bbox_to_anchor=(1.5, 1), loc = 'upper right')
