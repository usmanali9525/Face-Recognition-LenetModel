import cv2
import numpy as np
import face_recognition

input_image_path = r'Known/Usman Ali.jpeg'
input_image = face_recognition.load_image_file(input_image_path)

face_locations = face_recognition.face_locations(input_image)

if len(face_locations) == 0:

    print("No faces found in the input image.")

else:

    top, right, bottom, left = face_locations[0]
    face_image = input_image[top:bottom, left:right]
    face_image = cv2.resize(face_image, (256, 256))
    face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image_gray = cv2.equalizeHist(face_image_gray)

    cv2.imwrite('Usman Ali.jpeg', face_image_gray)

    cv2.waitKey(0)
