import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model(r'C:\Users\Admin\Documents\GitHub\Face-Recognition-LenetModel\Face Detection.h5')
class_names = {
    0: 'Alizar Waqar', 
    1: 'Gulfam Asghar', 
    2: 'Muhammad Arslan',
    3: 'Muhammad Shabbir', 
    4: 'Muhammad Yaseen', 
    5: 'Sufyan Ashraf',
    6: 'Talal Ahmad',
    7: 'Usman Ali'
}

def recognize_face(face):
    # Resize the face to match the input size of the recognition model
    resized_face = cv2.resize(face, (64, 64))
    
    prediction = model.predict(np.expand_dims(resized_face, axis=0))
    recognized_person_index = np.argmax(prediction)
    highest_probability = prediction[0][recognized_person_index]
    recognition_threshold = 0.70
    
    if highest_probability >= recognition_threshold:
        recognized_person_name = class_names.get(recognized_person_index)
        return recognized_person_name
    else:
        return "Person Not Recognized"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        recognized_person_name = recognize_face(face)
        
        print(f"Recognized Person: {recognized_person_name}")
        cv2.putText(frame, recognized_person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
