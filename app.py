import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("mask_detector.h5")

import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FACE_PROTO = os.path.join(BASE_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

st.title("😷 Face Mask Detector — Live Webcam")
st.write("Click START to begin real-time detection")

# --- Session state to control webcam ---
if "run" not in st.session_state:
    st.session_state.run = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ START Camera"):
        st.session_state.run = True
with col2:
    if st.button("⏹ STOP Camera"):
        st.session_state.run = False

# Placeholder where webcam frames will appear
frame_placeholder = st.empty()
result_placeholder = st.empty()

if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam!")
            break

        (h, w) = frame.shape[:2]

        # Detect faces
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX   = min(w - 1, endX)
            endY   = min(h - 1, endY)

            # Preprocess face
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Predict
            (mask, no_mask) = model.predict(face)[0]

            if mask > no_mask:
                label = "Mask"
                color = (0, 200, 0)       # green
                confidence_score = mask
            else:
                label = "No Mask"
                color = (0, 0, 220)       # red
                confidence_score = no_mask

            results.append(f"**{label}** — {confidence_score * 100:.1f}% confidence")

            # Draw box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            text = f"{label}: {confidence_score * 100:.1f}%"
            cv2.putText(
                frame, text,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Show results below the frame
        if results:
            result_placeholder.markdown("### Detection Result: " + " | ".join(results))
        else:
            result_placeholder.markdown("### No face detected — move closer to camera")

    cap.release()

else:
    frame_placeholder.markdown("### 👆 Press START to turn on webcam")