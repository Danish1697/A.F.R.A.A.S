import cv2
import os

user_name = str(input("Enter your name: "))
os.makedirs(f"dataset/{user_name}", exist_ok=True)

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print("Starting Webcam...")

for i in range(51):

    while True:

        ret, frame = camera.read()

        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:

            for x, y, w, h in faces:

                cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 255, 0),3)

                color_img = frame[y:y+h, x:x+w]

                cv2.imwrite(f"dataset/{user_name}/face{i}.jpg", color_img)

                print(f"Face {i} saved")

                break

            break

        cv2.imshow("Live Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting Webcam....")
            camera.release()
            cv2.destroyAllWindows()
            exit()

camera.release()
cv2.destroyAllWindows()

print("51 face images captured successfully!")