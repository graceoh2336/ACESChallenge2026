import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load models
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

classifier = tf.keras.models.load_model("../models/siren_classifier.keras")

#audio_file = "../data/test/notSiren/bbc_alarm.wav"
audio_file = "../data/test/siren/bbc_ambulance.wav"
#Load audio
waveform, sr = librosa.load(
    audio_file,
    sr=16000,
    mono=True
)

_, embeddings, _ = yamnet(waveform)

predictions = classifier.predict(embeddings.numpy(), verbose=0)

mean_probability = float(np.mean(predictions))
max_probability = np.max(predictions)

print(f"{audio_file} -> {mean_probability:.4f}")
print(f"{audio_file} -> {np.max(predictions):.4f}")

if max_probability > 0.9:
    print("Siren Detected")
else:
    print("No Siren Detected")