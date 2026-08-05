import numpy as np
import sounddevice as sd
import tensorflow as tf
from pathlib import Path
from audioUtils import get_embeddings, SAMPLE_RATE

WINDOW_SECONDS = 1.0 #Must be long enough for YAMNet (Around 0.96s minimum)
HOP_SECONDS = 0.25
THRESHOLD = 0.90
HITS_NEEDED = 3

print("Loading classifier...")
MODEL_PATH = (Path(__file__).resolve().parent.parent/ "models"/ "siren_classifier.keras")
classifier = tf.keras.models.load_model(MODEL_PATH)

window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
hop_samples = int(HOP_SECONDS * SAMPLE_RATE)
buffer = np.zeros(window_samples, dtype="float32")
 
consecutive_hits = 0

print("Listening... Press Ctrl+C to stop.")

while True:
    print("Recording...")

    chunk = sd.rec(hop_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    chunk = chunk.flatten()
 
    #Slide the window forward instead of using a lone short chunk
    buffer = np.concatenate([buffer[hop_samples:], chunk])

    # Generate embeddings
    embeddings = get_embeddings(buffer)

    # Predict
    predictions = classifier(embeddings, training=False).numpy()
    mean_probability = float(np.mean(predictions))
    max_probability = float(np.max(predictions))

    print(f"Mean={mean_probability:.3f}")
    print(f"Max={max_probability:.3f}")

    if max_probability > THRESHOLD:
        consecutive_hits += 1

    if consecutive_hits >= HITS_NEEDED:
        print("SIREN DETECTED")
        consecutive_hits = 0
