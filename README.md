# Facial Emotion-Based Music Recommendation System

A web app that detects your facial emotion from a photo and recommends songs
that match your mood — built as an academic/portfolio project.

## How it works

1. **Face detection** — OpenCV's Haar Cascade classifier locates a face in
   the uploaded photo or webcam capture.
2. **Emotion recognition** — A custom-trained Convolutional Neural Network
   (CNN), trained from scratch on the FER2013 dataset, classifies the face
   into one of 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
3. **Song recommendation** — The detected emotion is mapped to a target
   mood zone (valence/arousal), and the closest-matching songs from the
   Muse dataset (90,000+ Last.fm-tagged songs) are recommended.

## Tech stack

- **TensorFlow / Keras** — CNN model training and inference
- **OpenCV** — face detection (Haar Cascade)
- **Streamlit** — web app interface
- **Pandas / NumPy** — data handling for song recommendations

## Project structure

- `src/model_def.py` — CNN architecture (shared by train/test/app)
- `src/train_model.py` — Trains the CNN on FER2013
- `src/test_model.py` — Sanity-checks the saved model on a sample image
- `src/recommend.py` — Loads Muse dataset, maps emotion to songs
- `src/app.py` — Streamlit app (main entry point)
- `models/emotion_model.weights.h5` — Trained CNN weights
- `data/haarcascade_frontalface_default.xml` — Face detection classifier
- `data/fer2013/` — Training images (not included in repo, see below)
- `data/muse_v3.csv` — Song dataset (not included in repo, see below)
- `requirements.txt` — Python dependencies
- `README.md` — This file

## Model performance

The CNN was trained from scratch (not a pretrained model) on FER2013 for
15 epochs on CPU, achieving **~59% validation accuracy**. This is a
reasonable result for FER2013, which is a notoriously difficult dataset
even for state-of-the-art models (published benchmarks typically range
60-75%).

## Setup instructions

### 1. Clone the repo and set up a virtual environment
```bash
git clone https://github.com/keerthanakankanala/facial-emotion-based-music-recommender.git
cd facial-emotion-based-music-recommender
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** Requires Python 3.10+ (TensorFlow 2.20 is not compatible with
Python 3.9).

### 2. Download the datasets (not included in this repo due to size)

- **FER2013**: https://www.kaggle.com/datasets/msambare/fer2013
  Unzip into `data/fer2013/` (should contain `train/` and `test/` folders).

- **Muse dataset**: https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset
  Place `muse_v3.csv` into `data/`.

### 3. Train the model (or use the pre-trained weights already included)
```bash
python3 -m src.train_model
```

### 4. Run the app
```bash
python3 -m streamlit run src/app.py
```

## Limitations & notes

- Trained on a CPU with a modest CNN for feasibility — accuracy could be
  improved with a deeper architecture, more epochs, or a GPU.
- Emotion-to-mood mapping for song recommendation is hand-tuned, not
  scientifically validated.
- No external API dependencies (e.g. Spotify) are used — all
  recommendations come from the local Muse dataset for portability and to
  avoid exposing API credentials.

## Author

Keerthana K
