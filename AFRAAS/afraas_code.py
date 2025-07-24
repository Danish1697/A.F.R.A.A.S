import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Step 1: Define and verify the ImagesAttendance folder
script_dir = os.path.dirname(os.path.abspath(__file__))   # Current script directory
path = os.path.join(script_dir, 'ImagesAttendance')       # Full path to ImagesAttendance

# Create folder if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)
    print("✅ Created missing folder: ImagesAttendance")
    print("📂 Please add face images to the folder and run the script again.")
    exit()

# Load images
images = []
classNames = []
myList = os.listdir(path)

if len(myList) == 0:
    print("⚠️ No images found in ImagesAttendance folder. Please add face images and restart.")
    exit()

for cl in myList:
    img_path = os.path.join(path, cl)
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(cl)[0])  # Remove file extension

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
        else:
            print("⚠️ Warning: No face found in one of the images. Skipping.")
    return encodeList

def markAttendance(name):
    attendance_file = os.path.join(script_dir, 'Attendance.csv')
    
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Time,Date\n')

    with open(attendance_file, 'r+') as f:
        dataList = f.readlines()
        nameList = [line.split(',')[0] for line in dataList]
        if name not in nameList:
            now = datetime.now()
            timeStr = now.strftime('%H:%M:%S')
            dateStr = now.strftime('%Y-%m-%d')
            f.write(f'{name},{timeStr},{dateStr}\n')
            print(f"✅ Attendance marked for: {name}")

encodeListKnown = findEncodings(images)
print("✅ Encoding Complete. Starting webcam...")

# Step 2: Open webcam and start recognition
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from webcam.")
        break

    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Rescale

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('AFRAAS - Face Attendance System', img)
    if cv2.waitKey(1) == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()