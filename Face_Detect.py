#Libraries 
import numpy as np
import cv2	

#Including the xml file native to openCV since it seems to cause issues when not hardcoding the path to it
haarcascades_path = "."

def FacialDetector(imageURL, imgName, dest): 

	image = cv2.imread(imageURL) #Import the image
	greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier(haarcascades_path +'haarcascade_frontalface_alt.xml')

	#https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
	facesDetected = face_cascade.detectMultiScale(greyImage,1.05, 4)

	numsOfFaces = 0;
	for (column, row, width, height) in facesDetected:
		numsOfFaces += 1
		suffix = str(numsOfFaces)   
		aFace = np.array(image[row:row+height, column:column+width])
		aFace = cv2.resize(aFace, (150,150))
		cv2.imwrite(dest + imgName +"_"+ suffix + ".png",aFace); 


def FacialDetectorForModel(imageURL): 

	image = cv2.imread(imageURL) #Import the image
	greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier(haarcascades_path +'haarcascade_frontalface_alt.xml')

	#https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
	facesDetected = face_cascade.detectMultiScale(greyImage,1.05, 4)

	faces = []

	numsOfFaces = 0;
	for (column, row, width, height) in facesDetected:
		numsOfFaces += 1
		suffix = str(numsOfFaces)   
		aFace = np.array(image[row:row+height, column:column+width])
		aFace = cv2.resize(aFace, (150,150))
		faces.append(aFace)

	return faces
		



