#!/usr/bin/python

#This is the final file used to detect deep fakes
import sys
from Face_Detect import *
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

json_file_model = open('./Model/model.json', 'r') #Load the model framework via json file
model_info = json_file_model.read()
json_file_model.close()

deepfake_detector_model = model_from_json(model_info)
deepfake_detector_model.load_weights("./Model/08-0.12.hdf5") #Load the weights from the trained model (trained in Model_Creation.py)
#compile the model with the same loss, optimizer and monitoring metric as before 
deepfake_detector_model.compile(
    loss="binary_crossentropy",
    optimizer= keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"])


#Make sure there is an image path provided to evaluate 
if(len(sys.argv)>1):
	
	img_path = sys.argv[1]

	#Detect all the faces in the provided image 
	faces = FacialDetectorForModel(img_path) 
	

	#If faces are detected, have the CNN classify it 
	if(len(faces)>0):

		for face in faces:

			# Fit the image to dimensions accepted by model and scale the image values
			face_dim_for_model = np.expand_dims(face/255, axis=0)

			#Have the model predict the face's classification 
			prediction = deepfake_detector_model.predict(face_dim_for_model)[0]

			face_recoloured = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			plt.imshow(np.array(face_recoloured))

			if prediction<0.5:
				plt.title("Detected face is a deepfake.")
				plt.show()
				print(prediction)
			else:
				plt.title("Detected face is unaltered.")
				plt.show()
				print(prediction)


	else:
		print("No faces were detected.")

else:
	print("Please enter a path to the image as an arguement.")


