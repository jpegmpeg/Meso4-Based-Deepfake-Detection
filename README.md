# Deepfake Detection - Convolutional Neural Network


The archetecture of the CNN is based on Meso 4 (https://arxiv.org/abs/1809.00888)

Repository is reflects a project to impliment a CNN for combating facial forgery as generative AI becomes more prevalent 

##Execution

To run on commandline(while in ./): python3 DeepFakeDetector.py file_path_of_image_to_analyze

Note: The above should work as is if tensorflow is installed. I use a Mac with an M1 chip so I had to find a work around. I've included a file needed for the workaround in the ./Apple_M1_Extra_for_Tensor if needed. Furthermore, here is a link to a youtube video for the workaround if needed (https://www.youtube.com/watch?v=_CO-ND1FTOU). I ended up having to run my python through Jupyter Notebook, hence why the output doesn't directly save an image but rather plots it. 

To run on Mac M1 with above workaround, input into command line: 
	- conda activate tensorflow
	- jupyter notebook
	- Select the jupyter notebook "DF_Model_Detector.ipynb" I have provided

## Structure:

./Dataset
	- Contains subfolders Train and Test, which both contain deepfake_faces and real_faces. 7 Sample images are included in each of these folders to show a small sample of what the data I used to train and test the model looked like. 

./Model
	- Contains model weights at different epochs in training and the model architecture.

./Sample_Images
	-Contains the image used to produce the image in the figure, and other images if you want to play around with it.

./Detection_Of_DeepFake_Faces_Jesse_Mendoza.pdf
	-The report for the project. 

./Dataset_Processor.py
	- This python script help process the datasets to put them in the correct folder found in Dataset 

./Face_Detect.py
	- Includes the implementation of the Viola Jones algorithm

./haarcascade_frontalface_alt.xml
	- Contains the pretrained haarcascade model, it should come with OpenCV but I had issues so I downloaded the source code and provided a copy here so there would be no issues. 

./Model_Creation.py
	- Includes the code that creates and trains the neural network

./Model_Analysis.py
	- Includes the code for the computation of the analysis on the model predictions of the test set. Creates ROC, confusion matrix, and classification report. 

./DeepFakeDetector.py
	- The final algorithm, loads the model and weights, processes the image corresponding to the path passed in as an argument, extracts faces, makes predictions with the model, plots the predictions. 


