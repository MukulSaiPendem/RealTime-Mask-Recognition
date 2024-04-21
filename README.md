# Face Mask Recognition Project

## Introduction
This project uses deep learning and computer vision techniques to accurately detect facemasks on individuals in real-time. It features a model training component and a web application for model deployment.

## Project Structure
- **train.py**: This script handles the training of a deep learning model based on the MobileNetV2 architecture, adapted for facemask detection. It includes preprocessing steps like image data augmentation to enhance the model's robustness to variations in new images.
- **streamlit.py**: This Streamlit application provides a user interface for real-time facemask detection using a webcam. It integrates OpenCV for face detection and TensorFlow for mask detection based on the trained model.

### Detailed Breakdown
#### train.py
- **Image Preprocessing**: Uses `ImageDataGenerator` from Keras for real-time data augmentation, improving generalization by introducing random transformations on training images.
- **Model Setup**: Constructs a model using the MobileNetV2 architecture pre-trained on ImageNet, adding custom top layers to specifically suit the mask/no-mask binary classification.
- **Training Process**: Configures training parameters such as the optimizer, loss function, and learning rate. Executes training over several epochs, saving the best model based on validation accuracy.

#### streamlit.py
- **Streamlit Setup**: Initializes a Streamlit web interface for user interactions.
- **Video Processing**: Utilizes OpenCV to capture video frames from a webcam. Implements real-time face detection using a pre-trained deep learning model.
- **Mask Detection**: Each detected face is then processed through the trained facemask classifier to determine whether it is covered by a mask.
- **UI Display**: Results are dynamically displayed on the Streamlit interface, showing the video feed with bounding boxes around faces colored based on mask detection results (e.g., green for mask, red for no mask).

## Setup Instructions
### Prerequisites
Install Python 3.x and pip. Then, install the required packages:
```bash
pip install tensorflow keras opencv-python-headless streamlit imutils numpy
```

### Training the Model
To train the facemask detection model, place your dataset in the correct format and run:
```bash
python train.py
```
Make sure your dataset is structured with directories labeled 'mask' and 'no mask' for training images.

### Running the Streamlit Application
Launch the application with:
```bash
streamlit run streamlit.py
```
Navigate to the provided local URL in your web browser to interact with the application.

## Usage
Use the Streamlit interface to start your webcam and monitor real-time facemask detection. The application will display the video with live annotations showing whether individuals are wearing masks.

## Contributions
We encourage contributions to this project. Please fork the repository, make your changes, and submit a pull request for review.

## License
MIT License

Copyright (c) 2024 MukulSaiPendem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
