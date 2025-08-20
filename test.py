import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from time import sleep
from keras.models import load_model
CATEGORIES=["arul","charles","anto","none"]
model = tf.keras.models.load_model("CNN.model")
def prepare(file):
    IMG_SIZE = 150
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.equalizeHist(img_array)
    img_array = cv2.Canny(img_array, threshold1=3, threshold2=10)
    img_array = cv2.medianBlur(img_array,1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import cv2
import numpy as np
x=int(input("PRESS 1 TO TEST LEFT EAR 2 TO RIGHT EAR"))
if x==1:
    ear_cascade = cv2.CascadeClassifier('cascade.xml')
else:
    ear_cascade = cv2.CascadeClassifier('cascade1.xml')

print("AFTER EAR DETECTION, CLICK 'Q' KEY TO TEST IMAGE.")
cap = cv2.VideoCapture(0)

while 1:
  ret, img = cap.read()


  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ear = ear_cascade.detectMultiScale(gray, 1.3, 5)


  for (x,y,w,h) in ear:
      cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
      if cv2.waitKey(1) & 0XFF == ord('q'):
          Region = img[y:y + h, x:x +w]
          cv2.imwrite("10.jpg", Region)
          sleep(5)
          filename="10.jpg"
          prediction = model.predict(prepare(filename))
          prediction = list(prediction[0])
          print(prediction)
          l=CATEGORIES[prediction.index(max(prediction))]
          print(CATEGORIES[prediction.index(max(prediction))])
          break

  cv2.imshow('img',img)
  k = cv2.waitKey(30) & 0xff


cap.release()
cv2.destroyAllWindows()
