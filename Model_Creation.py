#Libraries 
#For Mac M1(as in my case): Run via conda enivornment to get tensorflow   
#	conda activate tensorflow
#	jupyter notebook

import os
import numpy as np
import cv2	

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

#Image input, resize it to 150, and include the 3 colour channels 
image_input = keras.Input(shape=(150, 150, 3), name = "image")

#3)
x = layers.Conv2D(filters = 8, kernel_size = (3, 3), padding = "same", activation="relu")(image_input)
x = layers.BatchNormalization()(x) #Normalize output
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #Downsample input by half (by taking max value in a 2x2 window)

#Convolutional layer 2 (8 nodes with 5x5 kernels (padding indluing to keep dimension) and use a ReLU activation function )
x = layers.Conv2D(filters = 8, kernel_size = (5, 5), padding = "same", activation="relu")(x)
x = layers.BatchNormalization()(x) #Normalize output
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #Downsample input by half (by taking max value in a 2x2 window)

#Convolutional layer 3 (16 nodes with 5x5 kernels (padding indluing to keep dimension) and use a ReLU activation function )
x = layers.Conv2D(filters = 16, kernel_size = (5, 5), padding = "same", activation="relu")(x)
x = layers.BatchNormalization()(x) #Normalize output
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #Downsample input by half (by taking max value in a 2x2 window)

#Convolutional layer 4 (16 nodes with 5x5 kernels (padding indluing to keep dimension) and use a ReLU activation function )
x = layers.Conv2D(filters = 16, kernel_size = (5, 5), padding = "same", activation="relu")(x)
x = layers.BatchNormalization()(x)#Normalize output
x = layers.MaxPooling2D(pool_size=(4, 4))(x)#Downsample input by quarter (by taking max value in a 4x4 window)

#Hidden dense layer
x = layers.Flatten()(x) #Flatten each input into singular dimension 
x = layers.Dropout(0.5)(x) #Drop half of inputs randomly (utilized in training)
x = layers.Dense(16)(x) #Changed from 16 since the input is s 150x150 rather than 256x256

#Output
x = layers.Dropout(0.5)(x) #Drop half of inputs randomly (utilized in training)
output = layers.Dense(1, activation = "sigmoid", name = "model_output")(x) #Turn the inputs all into a singular value via sigmoid for binary classification 

#Create the mode with the above architecure 
model = keras.Model(inputs=image_input, outputs=output, name="DF_Detection")
model.summary()

#Path to the training and test data. Since these datasets wont be included in the submission, these will not work without alteration 
training_data_path = "/Users/jessemendoza/Documents/School/Carleton/Year 2/Winter 2022/COMP 4102/Project/Dataset/Train"
testing_data_path = "/Users/jessemendoza/Documents/School/Carleton/Year 2/Winter 2022/COMP 4102/Project/Dataset/Test"

data_categories_train=os.listdir(training_data_path)
data_categories_test=os.listdir(testing_data_path)

#Data pipeline, with data augmentation for the training generator and use this to lower burden on memory
training_data_generator = ImageDataGenerator(
	rescale=1./255, #Rescale the inputs so values are (0,1)
	width_shift_range=0.2, #Data augmentation width shift
	height_shift_range=0.2, #Data augmentation height shift
	shear_range=0.2, #Data augmentation shear range
	brightness_range=[0.8,1.2], #Data augmentation brightness shift
	fill_mode='nearest', #Interpolation if needed
	validation_split=0.2) #Reserve part of the training data for validation

testing_data_generator = ImageDataGenerator(rescale=1./255)

train_flow = training_data_generator.flow_from_directory(
	training_data_path,
	color_mode="rgb",
	target_size=(150, 150),
	batch_size=32,
	shuffle=True, #Shuffle order of image presentation across epochs 
	class_mode='binary',
	subset='training')

validation_flow = training_data_generator.flow_from_directory(
	training_data_path,
	color_mode="rgb",
	target_size=(150, 150),
	batch_size=32,
	shuffle=True, #Shuffle order of image presentation across epochs
	class_mode='binary',
	subset='validation')

test_flow = testing_data_generator.flow_from_directory(
	testing_data_path,
	color_mode="rgb",
	target_size=(150, 150),
	batch_size=32,
	shuffle=False, #Do not shuffle order during testing, since if shuffle, it might not align with labels
	class_mode='binary')

#Compile the model with defitions of loss function to minimize and optimizer 
model.compile(
    loss="binary_crossentropy",
    optimizer= keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"])

#To attempt to avoid overfitting, If the accuracy stops improving over "patience" epochs, then take the model before those epochs. 
fitting_helper = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5) 

#Save the best models at checkpoints
checkpoint_filepath = "./Model/{epoch:02d}-{val_loss:.2f}.hdf5"

#Only save the best models (with weights) as accuracy is monitored and maxed
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#Train the model. Run a max of 30 epochs, using the model saving as a callback with the early stopping. 
training_history = model.fit_generator(train_flow,validation_data=validation_flow,epochs=30,verbose=1,callbacks=[model_checkpoint_callback,fitting_helper])
training_history.history

#Evaluation:
print("Evaluate:")

#Evaluate the performance of the model on the test images 
model_test_results = model.evaluate(test_flow)

#Save the framework of the model
model_as_json = model.to_json()
with open("./Model/model.json","w") as json_file:
  json_file.write(model_as_json)
