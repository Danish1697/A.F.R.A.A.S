import os
import face_recognition
import cv2
import numpy as np
import datetime

a= os.path.dirname(os.path.abspath(__file__))  # Current script directory
path = os.path.join(a, 'ImagesAttendance')  # Full path to ImagesAttendancea