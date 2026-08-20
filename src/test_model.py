"""
test_model.py
Sanity check: loads the saved model weights and runs a single prediction
on a real image from the test set, to confirm everything works end-to-end
before building the full Streamlit app.

Usage:
    python3 -m src.test_model
"""

import os
import random
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.model_def import build_model, IMG_SIZE, EMOTION_LABELS

WEIGHTS_PATH = "models/emotion_model.weights.h5"
TEST_DIR = "data/fer2013/test"

def pick_random_test_image():
    emotion = random.choice(os.listdir(TEST_DIR))
    emotion_dir = os.path.join(TEST_DIR, emotion)
    img_name = random.choice(os.listdir(emotion_dir))
    return os.path.join(emotion_dir, img_name), emotion

def main():
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    print("Model weights loaded successfully.\n")

    img_path, true_label = pick_random_test_image()
    print(f"Testing on: {img_path}")
    print(f"True label (from folder name): {true_label}\n")

    img = load_img(img_path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, 48, 48, 1)

    preds = model.predict(arr, verbose=0)[0]
    predicted_idx = np.argmax(preds)
    predicted_label = EMOTION_LABELS[predicted_idx]

    print("Predicted probabilities:")
    for label, prob in zip(EMOTION_LABELS, preds):
        print(f"  {label:10s}: {prob:.3f}")

    print(f"\nPredicted emotion: {predicted_label}")

if __name__ == "__main__":
    main()
