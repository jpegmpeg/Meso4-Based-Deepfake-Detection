import sys
from Face_Detect import *
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_curve,roc_auc_score,classification_report, confusion_matrix
import seaborn as sns

#Import the model and load its weights from the trained model. 
json_file_model = open('./Model/model.json', 'r')
model_info = json_file_model.read()
json_file_model.close()

deepfake_detector_model = model_from_json(model_info)
deepfake_detector_model.load_weights("./Model/08-0.12.hdf5")
deepfake_detector_model.compile(
    loss="binary_crossentropy",
    optimizer= keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"])

testing_data_path = "./Dataset/Test"
prediction_data_generator = ImageDataGenerator(rescale=1./255)

num_predictions = 10802 #images in test set

prediction_flow = prediction_data_generator.flow_from_directory(
    testing_data_path,
    color_mode="rgb",
    target_size=(150, 150),
    batch_size=1,
    shuffle=False, #Shuffle must be false or a different order of images are used each time it is called *
    class_mode='binary')

# *This is important here since if it were to shuffle the order, the orders would be different between prediction and true labels

#Have the model make predictions on the test images
predictions = deepfake_detector_model.predict(prediction_flow,steps = num_predictions)

#Have the true label of the images
true_labels = prediction_flow.classes

#Get ROC assessment values 
false_positive_rate , true_positive_rate , thresholds = roc_curve ( true_labels , predictions)

#Plot the ROC curve 
plt.plot(false_positive_rate,true_positive_rate) 
plt.axis([0,1,0,1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.show()  

print("AUC score: ")
print(roc_auc_score(true_labels,predictions) )  

#Use 0.5 as a threhold prediction between binary classes
prediction_with_classes = np.round(predictions)

#Make a confusion matrix of the predictions 
print('Confusion Matrix: ')
confusion_mapping = sns.heatmap(
    confusion_matrix(true_labels, prediction_with_classes), 
    annot=True,
    cmap='Blues')

confusion_mapping.set_xlabel('Predicted Values')
confusion_mapping.set_ylabel('Actual Values ')

Classifications = ['deep_faces', 'real_faces']

confusion_mapping.xaxis.set_ticklabels(Classifications)
confusion_mapping.yaxis.set_ticklabels(Classifications)

plt.show()

print('Classification Report')

print(classification_report(true_labels, prediction_with_classes, target_names=Classifications))


