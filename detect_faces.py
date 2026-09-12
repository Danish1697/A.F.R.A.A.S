import cv2
import os
import face_recognition as fr
import csv
from datetime import datetime

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load Image
known_image = fr.load_image_file("dataset/gg/face0.jpg")

known_encoding = fr.face_encodings(known_image)[0]

attendance_marked = False

print("Starting Webcam...")
while True:
    ret, frame = camera.read()
    if ret==False:
        break
    
    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = fr.face_locations(rgb_frame)
    
    # Get face encodings
    encodings = fr.face_encodings(rgb_frame, faces)
    
    # Go through each detected face
    for i in range(len(faces)):

        top, right, bottom, left = faces[i]

        result = fr.compare_faces([known_encoding],encodings[i])

        if result[0]:

            cv2.rectangle(frame,(left, top),(right, bottom),(0, 255, 0),3)

            cv2.putText(frame,"Danish",(left, top - 10),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),2)

            # Attendance
            if attendance_marked == False:

                with open("attendance.csv", "a", newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow(["Danish",datetime.now().strftime("%Y-%m-%d"),datetime.now().strftime("%H:%M:%S")])

                attendance_marked = True

                print("Danish Attendance Marked!")
    
    cv2.imshow('Live Attendance',frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        print("Quiting Webcam...")
        break
camera.release()
cv2.destroyAllWindows()