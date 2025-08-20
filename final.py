import face_recognition
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from time import sleep

# Load face recognition models and known faces
image_1 = face_recognition.load_image_file("5.jpg")
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]
image_2 = face_recognition.load_image_file("1.jpg")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]

known_face_encodings = [
    image_1_face_encoding,
    image_2_face_encoding,
    image_3_face_encoding,
]
known_face_names = ["THAJU","THIRU"]

# Load ear detection model
CATEGORIES = [" ","THAJU","THIRU"]
model = tf.keras.models.load_model("CNN.model")

def prepare(file):
    IMG_SIZE = 150
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.equalizeHist(img_array)
    img_array = cv2.Canny(img_array, threshold1=3, threshold2=10)
    img_array = cv2.medianBlur(img_array, 1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize ear cascade classifier based on user input
x = int(input("PRESS 1 TO TEST LEFT EAR 2 TO RIGHT EAR"))
if x == 1:
    ear_cascade = cv2.CascadeClassifier('cascade.xml')
else:
    ear_cascade = cv2.CascadeClassifier('cascade1.xml')

print("AFTER EAR DETECTION, CLICK 'Q' KEY TO TEST IMAGE.")

frame_count = 0
name = "Unknown"
rt=0
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection for 50 frames
    if frame_count < 150:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                frame_count = 150  # Set frame count to exit face detection loop
                # Draw face bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Display face detection result
        cv2.imshow('Face Detection', frame)
        cv2.waitKey(1)

    else:
        # Ear detection
        ear = ear_cascade.detectMultiScale(gray, 1.3, 5)

        # Print the number of ears detected
        print(f"Number of ears detected: {len(ear)}")

        for (x, y, w, h) in ear:
            # Draw ear bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                Region = frame[y:y + h, x:x + w]
                cv2.imwrite("ear.jpg", Region)
                sleep(5)
                filename = "ear.jpg"
                prediction = model.predict(prepare(filename))
                prediction = list(prediction[0])
                print(f"Ear prediction: {prediction}")
                ear_id = CATEGORIES[prediction.index(max(prediction))]
                
                print(f"Ear ID: {ear_id}")

                # Match face and ear IDs
                from time import sleep
                if ear_id == name:
                    print(f"Name: {name}")
                    print("Authentication Successful")
                    cv2.destroyAllWindows()
                    sleep(100)
                    break
                    
                else:
                    print("Retry")
                    rt=1
                    cv2.destroyAllWindows()
                    break
                

        # Display combined result
        if rt==1:
            cv2.destroyAllWindows()
            break
        cv2.imshow('Combined Result', frame)
        cv2.waitKey(1)

    frame_count += 1



# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()


