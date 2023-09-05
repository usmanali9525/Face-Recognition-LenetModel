import face_recognition
import cv2
import os

def get_known_faces(known_dir):
    known_encodings = []
    known_names = []

    for root, dirs, files in os.walk(known_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for file in os.listdir(dir_path):
                img_path = os.path.join(dir_path, file)
                img = cv2.imread(img_path)
                img_enc = face_recognition.face_encodings(img)[0]
                known_encodings.append(img_enc)
                known_names.append(dir_name)

    return known_encodings, known_names

def draw_box_and_name(frame, face_location, name):
    top, right, bottom, left = face_location
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(frame, name, (left + 2, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def main():
    known_dir = 'train'  # Directory containing subdirectories for each person

    # Set the similarity threshold
    threshold = 0.5

    # Compute and store known face encodings
    known_encodings, known_names = get_known_faces(known_dir)

    video_capture = cv2.VideoCapture(0)  # Open the camera (0 is usually the default camera)

    while True:
        ret, frame = video_capture.read()  # Read a frame from the camera feed

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            results = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown Person"

            # Calculate face recognition similarity scores
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = int(face_distances.argmin())
                if results[best_match_index] and face_distances[best_match_index] <= threshold:
                    name = known_names[best_match_index]
                    draw_box_and_name(frame, face_location, name)
                else:
                    draw_box_and_name(frame, face_location, name)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
