import face_recognition
import cv2
import numpy as np
import serial
import time
from threading import Timer


x = 0
y = 0
z = 0
name = ""
xyu = ""
top = 4
right = 4
bottom = 4
left = 4
image_1 = face_recognition.load_image_file("1.jpg")
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]
image_2 = face_recognition.load_image_file("2.jpg")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]
image_3 = face_recognition.load_image_file("3.jpg")
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]

known_face_encodings = [
    image_1_face_encoding,
    image_2_face_encoding,
    image_3_face_encoding,
]

known_face_names = ["ARUL", "CHARLIE","0"]
face_locations = []
face_encodings = []
face_names = []
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam
first_match_index = "9"

while True:
    # ret, frame = video_capture.read()
    _, frame = video_capture.read()  # Use the webcam frame
    process_this_frame = True

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    name = ""

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        print(len(face_encodings))
        first_match_index = 9
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            cv2.imwrite("int.jpg", frame)
            if True in matches:
                first_match_index = matches.index(True)
                print(first_match_index)
                name = known_face_names[first_match_index]
            face_names.append(name)
            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    face_names = []
    print(name)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
