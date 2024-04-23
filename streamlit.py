import streamlit as app_interface
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import requests
import tempfile

def download_model(url):
    r = requests.get(url, allow_redirects=True)
    if r.status_code == 200:
        # Save the file in a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.caffemodel' if '.caffemodel' in url else '.keras')
        temp_file.write(r.content)
        temp_file.flush()
        return temp_file.name
    return None

def analyze_face_mask_presence(video_frame, detection_network, classification_network):
    # Determine frame dimensions and prepare a blob
    frame_height, frame_width = video_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(video_frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Run blob through detection network to get facial detections
    detection_network.setInput(blob)
    face_detections = detection_network.forward()

    # Setup lists for storing detected faces and their locations
    detected_faces = []
    face_locations = []
    mask_predictions = []

    # Process each face detection
    for i in range(face_detections.shape[2]):
        confidence_level = face_detections[0, 0, i, 2]

        # Filter detections with low confidence
        if confidence_level > 0.5:
            bounding_box = face_detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            startX, startY, endX, endY = bounding_box.astype("int")

            # Adjust bounding box to be within frame dimensions
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(frame_width - 1, endX), min(frame_height - 1, endY)

            # Extract and preprocess face region
            face = video_frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            detected_faces.append(face)
            face_locations.append((startX, startY, endX, endY))

    # Predict mask presence if faces are detected
    if detected_faces:
        detected_faces = np.array(detected_faces, dtype="float32")
        mask_predictions = classification_network.predict(detected_faces, batch_size=32)

    return face_locations, mask_predictions

def main():

    # URLs of the model files on GitHub
    github_url_detection_model = 'https://github.com/your_repo/path/deploy.prototxt.txt'
    github_url_weights = 'https://github.com/MukulSaiPendem/RealTime-Mask-Recognition.git/res10_300x300_ssd_iter_140000.caffemodel'
    github_url_mask_model = 'https://github.com/MukulSaiPendem/RealTime-Mask-Recognition.git/mask_detector.keras'

    # Download models
    path_to_detection_model = download_model(github_url_detection_model)
    path_to_weights = download_model(github_url_weights)
    mask_detection_model_path = download_model(github_url_mask_model)

    # Load face detection model
    # path_to_detection_model = r"C:\Users\pende\DSEM\Capstone\deploy.prototxt.txt"
    # path_to_weights = r"C:\Users\pende\DSEM\Capstone\res10_300x300_ssd_iter_140000.caffemodel"
    face_detection_network = cv2.dnn.readNet(path_to_detection_model, path_to_weights)

    # Load the classification model for masks
    # mask_detection_model_path = './mask_detector.keras'
    mask_classification_network = load_model(mask_detection_model_path)

    app_interface.title('Real-Time Face Mask Detection')
    app_interface.write("Detects mask usage in real-time video feed.")

    # Initialize webcam
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    image_display = app_interface.image([])

    while True:
        success, video_frame = video_capture.read()
        if not success:
            continue

        video_frame = imutils.resize(video_frame, width=600)
        face_locations, mask_statuses = analyze_face_mask_presence(video_frame, face_detection_network, mask_classification_network)

        # Display results on each detected face
        for (box, status) in zip(face_locations, mask_statuses):
            startX, startY, endX, endY = box
            mask_probability, without_mask_probability = status
            label = "Mask" if mask_probability > without_mask_probability else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask_probability, without_mask_probability) * 100)

            cv2.putText(video_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(video_frame, (startX, startY), (endX, endY), color, 2)

        image_display.image(video_frame, channels='BGR')

    video_capture.release()

if __name__ == "__main__":
    main()
