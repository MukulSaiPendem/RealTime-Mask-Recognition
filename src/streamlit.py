import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch import nn

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjust the sizing based on output from last pool
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  # Adjust flattening based on pool output size
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load('best_model_cnn_deep.pth', map_location=device))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (255, 0, 0), 1: (0, 255, 0)}  # Red for no mask, Green for mask

# Load Haar Cascade for face detection
face_cascade_path = './haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    st.error("Failed to load face detection model!")
    st.stop()

st.title('Real-time Face Mask Detection')
st.write("""
         This application uses your webcam to detect if you are wearing a mask. 
         It utilizes a convolutional neural network (CNN) to make predictions in real-time.
         Please click on 'Start Camera' to begin mask detection.
         """)
run = st.button('Start Camera')
FRAME_WINDOW = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error opening video stream or file")

def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        pil_img = Image.fromarray(face_img)
        tensor = transform(pil_img)
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

        cv2.rectangle(image, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(image, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(image, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return image

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to grab frame")
        continue
    frame = detect_mask(frame)
    FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)

if not run and cap.isOpened():
    cap.release()
    cv2.destroyAllWindows()
    st.write('Stopped')
