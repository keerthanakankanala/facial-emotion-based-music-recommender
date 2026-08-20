"""
train_model.py
Trains the CNN defined in model_def.py on the FER2013 dataset and saves
the trained weights to models/emotion_model.weights.h5.

Usage:
    python3 -m src.train_model
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model_def import build_model, IMG_SIZE, EMOTION_LABELS

TRAIN_DIR = "data/fer2013/train"
TEST_DIR = "data/fer2013/test"
MODEL_OUT = "models/emotion_model.weights.h5"
BATCH_SIZE = 64
EPOCHS = 15  # keep modest for CPU training

def main():
    os.makedirs("models", exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTION_LABELS if all(
            os.path.isdir(os.path.join(TRAIN_DIR, c.lower())) for c in EMOTION_LABELS
        ) else None,
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=train_gen.class_indices.keys(),
        shuffle=False
    )

    print("Class indices (label -> index):", train_gen.class_indices)

    model = build_model()
    model.summary()

    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen
    )

    model.save_weights(MODEL_OUT)
    print(f"\nSaved trained weights to {MODEL_OUT}")

if __name__ == "__main__":
    main()
