
# Real-Time Mask Recognition 🎭

## Project Overview 🌟

This project implements a real-time face mask detection system using deep learning 🧠. It leverages a convolutional neural network to classify faces detected in a webcam feed as either 'with_mask' 😷 or 'without_mask' 😶. The system is built using PyTorch for model training and evaluation, and Streamlit for deploying a user-friendly interface.

## Features 🚀

- **Real-Time Detection**: Uses webcam feed for on-the-fly face mask detection 🎥.
- **Deep Learning Model**: Built on PyTorch, utilizing CNNs for high accuracy 📈.
- **Streamlit Application**: Provides an interactive web interface for easy usage 🖥️.

## Repository Structure 📂


RealTime-Mask-Recognition/
│
├── data/                         # Dataset and data related utilities 📦.
│   └── face-mask-dataset.zip     # Zipped dataset for training.
│
├── models/                       # Trained model weights and architecture 🔍.
│   ├── best_model.pth            # Trained model weights.
│   └── best_model_cnn_deep.pth   # Enhanced CNN model weights.
│
├── outputs/                      # Logs and output files from training sessions 📈.
│   └── metrics.json
│
├── src/                          # Source code for training and inference scripts 💻.
│   ├── Face_Mask_Detection_Training.ipynb   # Training Notebook.
│   └── streamlit.py                         # Stream app.
│
├── utils/                                   # Utility scripts for data manipulation and other tasks 🔧.
│   └── haarcascade_frontalface_default.xml
│
├── .gitignore                               # Specifies intentionally untracked files to ignore 🚫.
├── LICENSE                                  # License file to define the terms under which this project is shared.
├── README.md                     # Project description file 📝.
└── requirements.txt              # All necessary libraries for the project 📚.


## Installation and Setup 🛠️

To set up and run this project, follow these steps:

1. **Clone the repository**:
    ```
    git clone https://github.com/your-username/RealTime-Mask-Recognition.git
    cd RealTime-Mask-Recognition
    ```

2. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```

3. **Run the Streamlit application**:
    ```
    streamlit run src/app.py
    ```

## Usage 📖

To start the mask detection, simply run the Streamlit app and allow webcam access. The system will then detect and display the mask status in real-time.

## Contributions and Feedback 🤝

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](link-to-issues). If you want to contribute, please open a pull request.

## License 📄

This project is [MIT licensed](./LICENSE).

