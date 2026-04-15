# =============================================================================
# detect_mask.py — Real-Time Face Mask Detection via Webcam
# =============================================================================
# This script:
#   1. Loads the pre-trained CNN mask detector (mask_detector.h5)
#   2. Uses OpenCV's DNN-based face detector to locate faces in each frame
#   3. Predicts "Mask" or "No Mask" for every detected face
#   4. Draws colour-coded bounding boxes and confidence labels
#   5. Displays the annotated live feed — press 'q' to quit
#
# Run this AFTER training:  python detect_mask.py
# =============================================================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH       = "mask_detector.h5"

# OpenCV's DNN face detector (ships with the repo — download links below)
# Prototxt  : https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
# Caffemodel: https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel
FACE_PROTO       = "deploy.prototxt.txt"
FACE_MODEL       = "res10_300x300_ssd_iter_140000.caffemodel"

CONFIDENCE_THRESHOLD = 0.5   # minimum confidence to consider a detection valid
IMAGE_SIZE           = (224, 224)

# Colour palette (BGR format for OpenCV)
COLOR_MASK    = (0, 200, 0)    # green  → mask worn
COLOR_NO_MASK = (0, 0, 220)    # red    → no mask
COLOR_TEXT    = (255, 255, 255) # white  → label text


# =============================================================================
# STEP 1: Load models
# =============================================================================

def load_models():
    """
    Loads:
      - OpenCV's Caffe-based face detector  (detects face locations)
      - Keras mask classifier               (predicts mask vs no-mask)
    """
    print("[INFO] Loading face detector...")
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

    print("[INFO] Loading mask detector model...")
    mask_net = load_model(MODEL_PATH)

    return face_net, mask_net


# =============================================================================
# STEP 2: Detect faces in a single frame
# =============================================================================

def detect_faces(frame, face_net, confidence_threshold):
    """
    Passes the frame through OpenCV's SSD face detector.

    Args:
        frame               : BGR image (numpy array).
        face_net            : Loaded cv2.dnn network.
        confidence_threshold: Detections below this score are discarded.

    Returns:
        faces (list of np.ndarray): Cropped face ROIs ready for classification.
        locs  (list of tuples):     Corresponding (startX, startY, endX, endY) boxes.
    """
    (h, w) = frame.shape[:2]

    # Create a blob from the frame: resize to 300×300, apply mean subtraction
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)   # ImageNet mean for BGR channels
    )

    face_net.setInput(blob)
    detections = face_net.forward()   # shape: (1, 1, N, 7)

    faces = []
    locs  = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < confidence_threshold:
            continue   # skip weak detections

        # Bounding box is given as fractions of image dimensions → convert to pixels
        box    = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Clamp to frame boundaries to avoid out-of-bounds cropping
        startX = max(0, startX)
        startY = max(0, startY)
        endX   = min(w - 1, endX)
        endY   = min(h - 1, endY)

        # Extract the face ROI and preprocess it for MobileNetV2
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue   # skip empty ROIs (edge case near frame borders)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)      # convert BGR → RGB
        face = cv2.resize(face, IMAGE_SIZE)               # resize to 224×224
        face = img_to_array(face)                         # numpy array
        face = preprocess_input(face)                     # scale to [-1, 1]

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    return faces, locs


# =============================================================================
# STEP 3: Classify faces (mask / no mask)
# =============================================================================

def predict_masks(faces, mask_net):
    """
    Runs the Keras model on a batch of face ROIs.

    Args:
        faces    : List of preprocessed face arrays.
        mask_net : Loaded Keras model.

    Returns:
        preds (np.ndarray): Array of shape (N, 2) — [mask_prob, no_mask_prob] per face.
    """
    return mask_net.predict(np.array(faces), batch_size=32)


# =============================================================================
# STEP 4: Annotate the frame
# =============================================================================

def annotate_frame(frame, locs, preds):
    """
    Draws bounding boxes and prediction labels onto the frame in-place.

    Green box + "Mask XX%"    → mask detected
    Red box   + "No Mask XX%" → no mask detected
    """
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask_prob, no_mask_prob)    = pred

        # Determine the predicted class and pick display values
        if mask_prob > no_mask_prob:
            label       = "Mask"
            color       = COLOR_MASK
            confidence  = mask_prob
        else:
            label       = "No Mask"
            color       = COLOR_NO_MASK
            confidence  = no_mask_prob

        label_text = f"{label}: {confidence * 100:.1f}%"

        # --- Draw bounding box ---
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness=2)

        # --- Draw label background rectangle for readability ---
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2
        )
        label_y = max(startY, text_h + 10)   # don't draw above the frame
        cv2.rectangle(
            frame,
            (startX, label_y - text_h - 10),
            (startX + text_w + 4, label_y + baseline - 10),
            color, cv2.FILLED
        )

        # --- Draw label text ---
        cv2.putText(
            frame, label_text,
            (startX + 2, label_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=COLOR_TEXT,
            thickness=2
        )

    return frame


# =============================================================================
# MAIN — Real-time webcam loop
# =============================================================================

def main():
    face_net, mask_net = load_models()

    # Open the default webcam (index 0)
    # Change to cv2.VideoCapture("path/to/video.mp4") to run on a video file
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check if it is connected.")
        return

    print("[INFO] Starting real-time detection — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from webcam.")
            break

        # --- Detect & classify ---
        faces, locs = detect_faces(frame, face_net, CONFIDENCE_THRESHOLD)

        if faces:
            preds = predict_masks(faces, mask_net)
            frame = annotate_frame(frame, locs, preds)

        # --- Display ---
        cv2.imshow("Face Mask Detection  |  Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()