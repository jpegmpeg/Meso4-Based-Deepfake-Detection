#Want to create a dataset that has labelled real faces and deepfake faces: 
# /Dataset
# |_/Train
# 	|___/real_faces
#	|___/deep_faces 
# |_/Test
# 	|___/real_faces
#	|___/deep_faces 


#The filepaths referred to throughout the code is with reference to the downloaded datasets
#Since the datasets are not included in the submission, the below code will not run immediately
#This code is included to show the datapreparation 

from Face_Detect import *

import os
import shutil
import random
import cv2
import csv

#Path information on the dataset location 
target = "./Dataset/"
target_test = target + "Test/"
target_train = target + "Train/"

############## Preparing Deep Face Dataset ##################

#Extracting images from the FaceForensic dataset on Kaggle 
#https://www.kaggle.com/datasets/greatgamedota/faceforensics 

def copyDP(): 

	deepFake_Source = "./cropped_images/"

	target_test_dp = target_test + "deep_faces/"
	target_train_dp = target_train + "deep_faces/"

	deep_Entries = os.scandir(deepFake_Source)

	#Go through files and put 80% into the training dataset and 20% in the testing dataset 
	for dp_entry in deep_Entries:
		print(dp_entry.name)

		entryPath = deepFake_Source + dp_entry.name

		try: 

			filesList = os.listdir(entryPath)

			for dp_file in filesList: 

				filepath = entryPath + "/" + dp_file

				if(random.random() >0.2):
					#Copy in Training 80% of the time
				 	shutil.copy(filepath, target_train_dp+dp_entry.name+"_"+dp_file)
				else:
					#Copy in Testing 20% of the time 
				 	shutil.copy(filepath, target_test_dp+dp_entry.name+"_"+dp_file)

		except OSError as e:

			print('Not directory. Error: %s' % e)

############## Preparing Real Face Dataset ##################

#Extracting images from the LFW dataset of Kaggle that include labelled real faces 
#http://vis-www.cs.umass.edu/lfw/

def copyReal():

	real_Source = "./lfw/"

	target_test_r = target_test + "real_faces/"
	target_train_r = target_train + "real_faces/"

	real_Entries = os.scandir(real_Source)

	#Go through each image 
	for r_entry in real_Entries:

		print(r_entry.name)

		entryPath = real_Source + r_entry.name

		try: 

			filesList = os.listdir(entryPath)

			#For each image of a specific person, use the facial detection algorithm to extract faces in images with the proper size
			for r_file in filesList: 

				filepath = entryPath + "/" + r_file

				if(random.random() >0.2):
					#Copy in Training 80% of the time
				 	FacialDetector(filepath, r_entry.name, target_train_r) 

				else:
					#Copy in Testing 20% of the time 
				 	FacialDetector(filepath, r_entry.name, target_test_r) 

		except OSError as e:

			print('Not directory. Error: %s' % e)

#Extracting images of real faces from the UTK dataset, these real faces vary across age, photo quality, etc...
#https://susanqq.github.io/UTKFace/

def copyRealUTK():

	real_Source = "./UTKFace/"

	target_test_r = target_test + "real_faces/"
	target_train_r = target_train + "real_faces/"

	filesList = os.listdir(real_Source)

	#Go through each image 
	for r_file in filesList: 

		filepath = real_Source + "/" + r_file

		image = cv2.imread(filepath) #Import the image

		aFace = cv2.resize(image, (150,150))

		if(random.random() >0.2):
			#Copy in Training 80% of the time
			cv2.imwrite(target_train_r + r_file + ".png",aFace); 

		else:
			#Copy in Testing 20% of the time 
			cv2.imwrite(target_test_r + r_file + ".png",aFace); 

#Code for extracting and reformating images of real faces from the Deepfake Facebook challenge 
#https://www.kaggle.com/datasets/dagnelies/deepfake-faces

def copyRealFBChallenge():

	path_to_data = "./metadata.csv"

	real_Source = "./faces_224/"

	target_test_r = target_test + "real_faces/"
	target_train_r = target_train + "real_faces/"

	#Open up the provided CSV to know whether an image is a Deepfake or real
	with open(path_to_data) as csv_obj:
		Real_photos_names = [ (photo[0].split(".")[0] + ".jpg") for photo in csv.reader(csv_obj) if (photo[3]=="REAL")]

	for r_file in Real_photos_names: 

		filepath = real_Source + r_file

		image = cv2.imread(filepath) #Import the image

		aFace = cv2.resize(image, (150,150))#Resize for input into nerual net

		if(random.random() >0.2):
			#Copy in Training 80% of the time
			cv2.imwrite(target_train_r + r_file + ".png",aFace); 

		else:
			#Copy in Testing 20% of the time 
			cv2.imwrite(target_test_r + r_file + ".png",aFace); 

#Code for extracting and reformating images of deepfake faces from the Deepfake Facebook challenge 
#https://www.kaggle.com/datasets/dagnelies/deepfake-faces
def copyDFFBChallenge():

	path_to_data = "./metadata.csv"

	real_Source = "./faces_224/"

	target_test_dp = target_test + "deep_faces/"
	target_train_dp = target_train + "deep_faces/"

	#Open up the provided CSV to know whether an image is a Deepfake or real
	with open(path_to_data) as csv_obj:
		Fake_photos_names = [ (photo[0].split(".")[0] + ".jpg") for photo in csv.reader(csv_obj) if (photo[3]=="FAKE")]

	#Trying to make the amount deepfake and real roughly the same (In terms of the end number of images)
	#^ is the reason why I only cycle through 22901 images
	for i in range(0,22900):

		filepath = real_Source + Fake_photos_names[i]

		image = cv2.imread(filepath) #Import the image

		aFace = cv2.resize(image, (150,150)) #Resize for input into nerual net

		if(random.random() >0.05):
			#Copy in Training 95% of the time
			cv2.imwrite(target_train_dp + Fake_photos_names[i] + ".png",aFace); 

		else:
			#Copy in Testing 5% of the time 
			cv2.imwrite(target_test_dp + Fake_photos_names[i] + ".png",aFace); 	

############## Calls to the above functions to produce the dataset #############################

#copyDP()

#copyReal()

#copyRealUTK()

#copyRealFBChallenge()

#copyDFFBChallenge()

####################### End Counts of Images stored for the dataset #############################

#Train

	#Deepfake: 38,570
	#Real: 38,571

#Test

	#Deepdake: 5,403
	#Real: 5,401



