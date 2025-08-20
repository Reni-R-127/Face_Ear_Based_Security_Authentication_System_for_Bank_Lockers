import cv2
import os
import numpy
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
from skimage import io, color
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pickle
from skimage import io, color
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

IMG_SIZE = 150
file_list = []
class_list = []
training_data = []
print('STARTING TRAINING!')



def perform_q_learning(image_path, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1):
    original_image = io.imread(image_path)
    gray_image = color.rgb2gray(original_image)

    # Initialize Q-values for each pixel
    Q_values = np.zeros_like(gray_image)

    # Q-learning training loop
    for episode in range(num_episodes):
        # Randomly select a starting point (pixel)
        start_pixel = (np.random.randint(0, gray_image.shape[0]),
                       np.random.randint(0, gray_image.shape[1]))

        # Q-learning episode
        current_pixel = start_pixel
        while True:
            # Select action using epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1])  # 0: do not segment, 1: segment
            else:
                action = np.argmax(Q_values[current_pixel])

            # Take action
            if action == 1:
                # Segment the pixel (you might need to adapt this based on your segmentation method)
                # For simplicity, we set the pixel value to 1 (segmented)
                Q_values[current_pixel] += alpha * (1 + gamma * np.max(Q_values) - Q_values[current_pixel])
            else:
                # Do not segment the pixel
                Q_values[current_pixel] += alpha * (gamma * np.max(Q_values) - Q_values[current_pixel])

            # Move to the next pixel (you might need to adapt this based on your movement strategy)
            next_pixel = (current_pixel[0] + 1, current_pixel[1])
            if next_pixel[0] >= gray_image.shape[0]:
                break  # End the episode if we reach the bottom of the image

            current_pixel = next_pixel

    # Threshold the Q-values to obtain the segmentation mask
    segmentation_mask = (Q_values > np.mean(Q_values))

    return segmentation_mask


def getImageId():
    paths = [os.path.join('images\\samples', i) for i in os.listdir('images\\samples')]
    for pathImage in paths:
        try:
            imageFace = cv2.imread(pathImage)
            imageConverted = cv2.cvtColor(imageFace, cv2.COLOR_BGR2GRAY)
            img_array = cv2.equalizeHist(imageConverted)
            img_array = cv2.Canny(img_array, threshold1=3, threshold2=10)
            img_array = cv2.medianBlur(img_array,1)
            img = perform_q_learning(pathImage)
            
            class_num=int(os.path.split(pathImage)[-1].split('.')[1])
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
                pass
        
    

getImageId()
random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X*3).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y=np.array(y*3)
# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

print('SUCCESSFUL TRAINING!')
from time import sleep
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
sleep(5)
# Opening the files about data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# normalizing data (a pixel goes from 0 to 255)
X = X/255.0

# Building the model
# Building the model
model = Sequential()
# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# The output layer with 13 neurons, for 13 classes
model.add(Dense(4))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(X, y, batch_size=32, epochs=5, validation_split=0.2)

# Saving the model

print("Saved model to disk")

model.save('CNN.model')

# Printing a graph showing the accuracy changes during the training phase
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
acc=np.array(acc)
val_acc=np.array(val_acc)
loss=np.array(loss)
val_loss=np.array(val_loss)
epochs_range = range(5)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
vgg_y_pred =  model.predict_generator(X)
y_pred_array=np.array(vgg_y_pred)

y_g=[]


print(y)
print(y_pred_array)
yt=[]
for xt in y_pred_array:
  yt.append(xt.tolist().index(max(xt)))
print(yt)





from sklearn.metrics import classification_report

print('\nClassification Report\n')
print(classification_report(y, yt))
confusion_mtx = confusion_matrix(y, yt)


# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

from sklearn import metrics
acc=(metrics.accuracy_score(y,yt)*100)
print("Accuracy is:",acc)
cm1 = metrics.confusion_matrix(y,yt)

total1=sum(sum(cm1))


sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

y_test_bin = label_binarize(y, classes=[0, 1, 2, 3])
y_pred_bin = label_binarize(yt, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

from itertools import cycle 
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-class Classification')
plt.legend(loc="lower right")
plt.show()

