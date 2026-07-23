import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.25

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("Loading classifier...")

MODEL_PATH = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "siren_classifier.keras"
)

classifier = tf.keras.models.load_model(
    MODEL_PATH
)


consecutive_hits = 0
print("Listening... Press Ctrl+C to stop.")

while True:
    print("Recording...")

    # Record 1 second
    audio = sd.rec(
        int(CHUNK_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"

    )

    sd.wait()

    waveform = audio.flatten()

    # Generate embeddings
    _, embeddings, _ = yamnet(waveform)

    # Predict
    predictions = classifier(
        embeddings,
        training=False
    ).numpy()

    mean_probability = float(np.mean(predictions))

    max_probability = float(np.max(predictions))

    print(
        f"Mean={mean_probability:.3f} "
        f"Max={max_probability:.3f}"
    )

    if max_probability > 0.90:
        consecutive_hits += 1


    if consecutive_hits >= 3:
        print("SIREN DETECTED")
