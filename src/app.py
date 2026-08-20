"""
app.py
Streamlit web app: captures a photo from your webcam, detects your face,
predicts your emotion using the trained CNN, and recommends songs that
match your mood using the Muse dataset.

Usage:
    python3 -m streamlit run src/app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from src.model_def import build_model, IMG_SIZE, EMOTION_LABELS
from src.recommend import load_dataset, recommend_songs

WEIGHTS_PATH = "models/emotion_model.weights.h5"
CASCADE_PATH = "data/haarcascade_frontalface_default.xml"


@st.cache_resource
def get_model():
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    return model


@st.cache_resource
def get_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)


@st.cache_data
def get_song_data():
    return load_dataset()


def detect_and_crop_face(image_np, detector):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face_resized


def predict_emotion(face_img, model):
    arr = face_img.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return EMOTION_LABELS[idx], preds


def main():
    st.set_page_config(page_title="Emotion-Based Music Recommender", page_icon="🎵")
    st.title("🎵 Facial Emotion-Based Music Recommender")
    st.write("Take a photo with your webcam and get song recommendations that match your mood.")

    model = get_model()
    detector = get_face_detector()
    song_df = get_song_data()

    input_mode = st.radio("Choose input method:", ["Upload a photo", "Use webcam"])

    if input_mode == "Upload a photo":
        photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    else:
        photo = st.camera_input("Take a photo")

    if photo is not None:
        image = Image.open(photo).convert("RGB")
        image_np = np.array(image)

        face_img = detect_and_crop_face(image_np, detector)

        if face_img is None:
            st.warning("No face detected. Try again with better lighting, facing the camera directly.")
            return

        emotion, probs = predict_emotion(face_img, model)

        st.subheader(f"Detected emotion: **{emotion}**")

        with st.expander("See prediction confidence"):
            for label, prob in zip(EMOTION_LABELS, probs):
                st.write(f"{label}: {prob:.1%}")

        st.subheader("Recommended songs for your mood:")
        songs = recommend_songs(emotion, song_df, n=5)
        for song in songs:
            st.markdown(f"**{song['track']}** by {song['artist']} ({song['genre']})")
            st.markdown(f"[Listen on Last.fm]({song['lastfm_url']})")
            st.divider()


if __name__ == "__main__":
    main()
