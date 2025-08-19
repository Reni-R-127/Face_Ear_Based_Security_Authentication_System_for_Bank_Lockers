import cv2
import numpy as np
x=int(input("PRESS 1 TO TRAIN LEFT EAR 2 TO RIGHT EAR"))
if x==1:
    ear_cascade = cv2.CascadeClassifier('cascade.xml')
else:
    ear_cascade = cv2.CascadeClassifier('cascade1.xml')
faceName = input('ENTER THE NAME OF THE FACE OWNER: ')
faceId = input('ENTER THE UNIQUE ID FOR THIS EAR: ')
sample=0
sampleNumber=25
print("AFTER EAR DETECTION, CLICK 'Q' KEY TO SAVE IMAGE.")
cap = cv2.VideoCapture(0)

while 1:
  ret, img = cap.read()


  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ear = ear_cascade.detectMultiScale(gray, 1.3, 5)


  for (x,y,w,h) in ear:
      cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
      if cv2.waitKey(1) & 0XFF == ord('q'):
          Region = img[y:y + h, x:x +w]
          cv2.imwrite("images\\samples\\" + str(faceName) + "." + str(faceId) + "." + str(sample) + ".jpg", Region)
          sample += 1
          print("PHOTO " + str(sample) + " SUCCESSFULLY CAPTURED")
  if (sample >= sampleNumber):
            break  
          
      

  cv2.imshow('img',img)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
      break


cap.release()
cv2.destroyAllWindows()