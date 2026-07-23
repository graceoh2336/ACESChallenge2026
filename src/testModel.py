import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

#classifier = tf.keras.models.load_model("../models/siren_classifier.keras")
classifier = tf.keras.models.load_model("/home/gohallor/ACES/Code/YamNetTrial/models/siren_classifier.keras")

for wav_file in Path("../data/test/siren").rglob("*.wav"):

    waveform, sr = librosa.load(
        wav_file,
        sr=16000,
        mono=True
    )

    _, embeddings, _ = yamnet(waveform)

    predictions = classifier.predict(
        embeddings.numpy(),
        verbose=0
    )

    mean_probability = float(np.mean(predictions))

    max_probability = float(np.max(predictions))

    print(
        f"{wav_file.name:40} "
        f"{mean_probability:.4f}"
        f"{max_probability:.4f}"
    )
