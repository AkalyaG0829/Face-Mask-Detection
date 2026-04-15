# =============================================================================
# train_model.py — Face Mask Detection: CNN Training Script
# =============================================================================
# This script handles:
#   1. Loading and preprocessing the dataset
#   2. Building the CNN model with Keras
#   3. Training and evaluating the model
#   4. Plotting accuracy/loss graphs
#   5. Saving the trained model
#
# Dataset expected structure:
#   dataset/
#     with_mask/      ← images of people wearing masks
#     without_mask/   ← images of people without masks
#
# Download a dataset from Kaggle:
#   https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    AveragePooling2D, Dropout, Flatten, Dense, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

import cv2

# =============================================================================
# CONFIGURATION — Adjust these settings as needed
# =============================================================================

DATASET_PATH   = "dataset"         # Root folder containing with_mask/ & without_mask/
MODEL_PATH     = "mask_detector.h5"  # Output path to save the trained model
IMAGE_SIZE     = (224, 224)        # MobileNetV2 expects 224x224 input
BATCH_SIZE     = 32
EPOCHS         = 20
LEARNING_RATE  = 1e-4
TEST_SPLIT     = 0.20              # 20% data reserved for testing
RANDOM_SEED    = 42

# =============================================================================
# STEP 1: Load and preprocess the dataset
# =============================================================================

def load_dataset(dataset_path):
    """
    Walks through dataset subfolders, loads each image, resizes it,
    normalises pixel values, and records its class label.

    Returns:
        data   (np.ndarray): Array of preprocessed images.
        labels (list):       Corresponding class labels (folder names).
    """
    print("[INFO] Loading images from dataset...")
    data   = []
    labels = []

    # Each subfolder name becomes the class label
    classes = sorted(os.listdir(dataset_path))

    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue  # skip any stray files at the root level

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load image, resize, convert to array
                image = load_img(img_path, target_size=IMAGE_SIZE)
                image = img_to_array(image)

                # preprocess_input scales pixels to [-1, 1] for MobileNetV2
                image = preprocess_input(image)

                data.append(image)
                labels.append(class_name)
            except Exception as e:
                print(f"  [WARNING] Could not load {img_path}: {e}")

    print(f"[INFO] Loaded {len(data)} images across {len(classes)} classes: {classes}")
    return np.array(data, dtype="float32"), labels


def encode_labels(labels):
    """
    One-hot encodes string labels using sklearn's LabelBinarizer.

    Returns:
        labels_encoded (np.ndarray): One-hot encoded label matrix.
        lb             (LabelBinarizer): Fitted binarizer (needed for class names later).
    """
    lb = LabelBinarizer()
    labels_encoded = lb.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded)  # ensure shape (N, num_classes)
    print(f"[INFO] Classes detected: {lb.classes_}")
    return labels_encoded, lb


# =============================================================================
# STEP 2: Build the CNN model (Transfer Learning with MobileNetV2)
# =============================================================================

def build_model(num_classes=2):
    """
    Constructs a transfer-learning model:
      - Base: MobileNetV2 (pretrained on ImageNet, top layers excluded)
      - Head: AveragePooling → Flatten → Dense(128, ReLU) → Dropout → Dense(2, Softmax)

    The base is frozen initially so only the custom head is trained.
    Fine-tuning the base in a second pass can further improve accuracy.

    Returns:
        model (keras.Model): Compiled model ready for training.
    """
    print("[INFO] Building model...")

    # Load MobileNetV2 without its classifier head
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,              # exclude ImageNet's 1000-class head
        input_tensor=Input(shape=(224, 224, 3))
    )

    # Freeze all base layers — we won't update their weights during phase 1
    base_model.trainable = False

    # --- Custom classification head ---
    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)              # regularisation to reduce overfitting
    head = Dense(num_classes, activation="softmax")(head)  # probability per class

    model = Model(inputs=base_model.input, outputs=head)

    # Compile with Adam optimizer and categorical cross-entropy loss
    optimizer = Adam(learning_rate=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    print(model.summary())
    return model


# =============================================================================
# STEP 3: Data Augmentation
# =============================================================================

def build_augmentor():
    """
    Creates a Keras ImageDataGenerator that applies random transformations
    to training images on-the-fly. This artificially enlarges the dataset
    and helps the model generalise better.

    Augmentations applied:
        - Random rotation (±20°)
        - Width / height shifts (up to 20%)
        - Horizontal flip
        - Zoom in/out (up to 15%)
        - Shear (up to 15%)
    """
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )


# =============================================================================
# STEP 4: Plot accuracy and loss curves
# =============================================================================

def plot_training(history, epochs):
    """
    Saves two side-by-side plots:
      - Left:  Training vs. Validation Accuracy
      - Right: Training vs. Validation Loss

    A growing gap between train and validation curves signals overfitting.
    """
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(1, epochs + 1)

    # Accuracy plot
    axes[0].plot(x, history.history["accuracy"],     label="Train Accuracy",      linewidth=2)
    axes[0].plot(x, history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss plot
    axes[1].plot(x, history.history["loss"],     label="Train Loss",      linewidth=2)
    axes[1].plot(x, history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_plot.png", dpi=150)
    print("[INFO] Training plot saved as training_plot.png")
    plt.show()


# =============================================================================
# MAIN — Orchestrates all steps
# =============================================================================

def main():
    # --- Load data ---
    data, labels = load_dataset(DATASET_PATH)
    labels_encoded, lb = encode_labels(labels)

    # --- Train / test split ---
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels_encoded,
        test_size=TEST_SPLIT,
        stratify=labels_encoded,   # preserve class distribution in both splits
        random_state=RANDOM_SEED
    )
    print(f"[INFO] Training samples: {len(trainX)} | Test samples: {len(testX)}")

    # --- Build model & augmentor ---
    model   = build_model(num_classes=len(lb.classes_))
    aug     = build_augmentor()

    # --- Train ---
    print("[INFO] Training model — this may take a few minutes...")
    history = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BATCH_SIZE,
        epochs=EPOCHS
    )

    # --- Evaluate ---
    print("[INFO] Evaluating model on test set...")
    predY = model.predict(testX, batch_size=BATCH_SIZE)
    predY = np.argmax(predY, axis=1)   # convert probabilities → class indices

    print(classification_report(
        np.argmax(testY, axis=1),
        predY,
        target_names=lb.classes_
    ))

    # --- Save model ---
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    # --- Plot ---
    plot_training(history, EPOCHS)


if __name__ == "__main__":
    main()