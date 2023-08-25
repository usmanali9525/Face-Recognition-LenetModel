# Face Recognition Using Python Library

This repository contains Python scripts for performing face recognition using the face_recognition library and OpenCV. The project consists of two main scripts: `FaceRecognition.ipynb` and `FaceRecognitionImage.ipynb`. The first script captures real-time video from the default camera and performs face recognition on the detected faces against a set of known faces. The second script focuses on performing face recognition on images in the "Unknown" directory against known faces in the "Known" directory.

## Prerequisites

Before running the scripts, make sure you have the following prerequisites installed:

- Python 3.x
- face_recognition library (`pip install face_recognition`)
- OpenCV (`pip install opencv-python`)

## Usage

### FaceRecognition.ipynb

This script captures real-time video from the default camera and performs face recognition on detected faces against a set of known faces. It displays bounding boxes and names of recognized faces.

To run the script:

1. Place images of known persons in the "Known" directory.
2. Run the `FaceRecognition.ipynb` notebook.
3. Press 'q' to exit the video feed.

### FaceRecognitionImage.ipynb

This script performs face recognition on images in the "Unknown" directory against known faces in the "Known" directory. It processes images and recognizes names.

To run the script:

1. Place images of known persons in the "Known" directory.
2. Place images for face recognition in the "Unknown" directory.
3. Run the `FaceRecognitionImage.ipynb` notebook.

## Folder Structure

- `Known`: This directory should contain images of known individuals for training.
- `Unknown`: This directory should contain images for which you want to perform face recognition.

## Contributing

Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request.
