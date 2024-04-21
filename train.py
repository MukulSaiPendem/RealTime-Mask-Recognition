# Required libraries and packages
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.preprocessing.image as image_processing
import tensorflow.keras.applications as applications
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import os

# Define learning parameters
learning_rate_initial = 0.0001
training_epochs = 20
batch_size = 32

# Set dataset directory and categories
image_directory = r"C:\Users\pende\DSEM\Capstone\facemask"
image_labels = ["with_mask", "without_mask"]

# Load dataset
print("[INFO] Starting image loading...")
images = []
classes = []

for label in image_labels:
    folder_path = os.path.join(image_directory, label)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        img = image_processing.load_img(file_path, target_size=(224, 224))
        img = image_processing.img_to_array(img)
        img = applications.mobilenet_v2.preprocess_input(img)
        images.append(img)
        classes.append(label)

# Encode class labels
encoder = LabelBinarizer()
classes_encoded = encoder.fit_transform(classes)
classes_encoded = image_processing.to_categorical(classes_encoded)

images = np.array(images, dtype="float32")
classes_encoded = np.array(classes_encoded)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(images, classes_encoded, test_size=0.20, stratify=classes_encoded, random_state=42)

# Augment data
augmentor = image_processing.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Setup the base MobileNetV2 model
base_model = applications.MobileNetV2(weights="imagenet", include_top=False, input_tensor=layers.Input(shape=(224, 224, 3)))

# Build the top layers for our model
top_model = base_model.output
top_model = layers.AveragePooling2D(pool_size=(7, 7))(top_model)
top_model = layers.Flatten(name="flatten")(top_model)
top_model = layers.Dense(128, activation="relu")(top_model)
top_model = layers.Dropout(0.5)(top_model)
top_model = layers.Dense(2, activation="softmax")(top_model)

# Combine models
complete_model = models.Model(inputs=base_model.input, outputs=top_model)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
print("[INFO] Compiling the full model...")
optimizer = optimizers.Adam(learning_rate=learning_rate_initial)
complete_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model
print("[INFO] Starting training...")
training_history = complete_model.fit(
    augmentor.flow(X_train, Y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_test, Y_test),
    validation_steps=len(X_test) // batch_size,
    epochs=training_epochs
)

# Evaluate the model
print("[INFO] Model evaluation...")
predictions = complete_model.predict(X_test, batch_size=batch_size)
predictions = np.argmax(predictions, axis=1)

# Classification results
print(classification_report(Y_test.argmax(axis=1), predictions, target_names=encoder.classes_))

# Save the model
print("[INFO] Saving the model...")
complete_model.save("mask_detection_model.h5", save_format="h5")

# Plotting training results
epochs_range = training_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs_range), training_history.history["loss"], label="Training Loss")
plt.plot(np.arange(0, epochs_range), training_history.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, epochs_range), training_history.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, epochs_range), training_history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Loss and Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("training_results.png")
