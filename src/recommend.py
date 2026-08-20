"""
recommend.py
Loads the Muse dataset (data/muse_v3.csv) and recommends songs based on a
detected emotion, using valence (positivity) and arousal (energy) scores.

Usage (standalone test):
    python3 -m src.recommend
"""

import pandas as pd
import numpy as np

MUSE_PATH = "data/muse_v3.csv"

EMOTION_TARGETS = {
    "Angry":    (3.0, 6.5),
    "Disgust":  (3.0, 5.0),
    "Fear":     (3.0, 6.0),
    "Happy":    (7.0, 6.0),
    "Sad":      (3.0, 3.0),
    "Surprise": (6.5, 6.5),
    "Neutral":  (5.0, 4.0),
}


def load_dataset(path=MUSE_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["valence_tags", "arousal_tags", "track", "artist"])
    return df


def recommend_songs(emotion, df=None, n=5):
    """Return n songs whose mood best matches the given emotion."""
    if df is None:
        df = load_dataset()

    if emotion not in EMOTION_TARGETS:
        raise ValueError(f"Unknown emotion: {emotion}")

    target_valence, target_arousal = EMOTION_TARGETS[emotion]

    distance = np.sqrt(
        (df["valence_tags"] - target_valence) ** 2
        + (df["arousal_tags"] - target_arousal) ** 2
    )
    df = df.assign(distance=distance)

    closest = df.nsmallest(50, "distance")
    picks = closest.sample(n=min(n, len(closest)))

    return picks[["track", "artist", "genre", "lastfm_url"]].to_dict("records")


if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded {len(df)} songs from {MUSE_PATH}\n")

    for emotion in EMOTION_TARGETS:
        print(f"--- Recommendations for {emotion} ---")
        for song in recommend_songs(emotion, df, n=3):
            print(f"  {song['track']} by {song['artist']} ({song['genre']})")
        print()
