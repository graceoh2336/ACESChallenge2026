import numpy as np
import tensorflow as tf
from pathlib import Path
from audioUtils import load_waveform, get_embeddings

# Load model
classifier = tf.keras.models.load_model(Path(__file__).resolve().parent.parent / "models" / "siren_classifier.keras")

#audio_file = "../dataset/testData/notSiren/bbc_alarm.wav"
audio_file = "../dataset/testData/siren/bbc_ambulance.wav"

#Load audio
waveform = load_waveform(audio_file)
embeddings = get_embeddings(waveform)
predictions = classifier.predict(embeddings, verbose=0)

mean_probability = float(np.mean(predictions))
max_probability = float(np.max(predictions))

print(f"{audio_file} -> {mean_probability:.4f}")
print(f"{audio_file} -> {np.max(predictions):.4f}")

if max_probability > 0.9:
    print("Siren Detected")
else:
    print("No Siren Detected")
