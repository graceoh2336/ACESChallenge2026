import librosa
import numpy as np
import tensorflow_hub as hub
from pathlib import Path

#To create embeddings (a vector of numbers describing the audio). Convert the audio in to machine learning features using YamNet and save model

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

#Create datasets
X = [] #What the model learns from (Large nums)
y = [] # he correct answer the model should learn to predict (1 or 0)

# Sirens
for wav_file in Path("../dataset/sirens").glob("*.wav"):
    waveform, sr = librosa.load(  #Sample Rate: Audio samples recorded per second
        wav_file,
        sr=16000, #YamNet automatically resamples the audio to16kHz
        mono=True #Combines all channels into one, as YamNet ants
    )

    #Returns score, embeddings, spectrogram but ignoring score & spectrogram.
    #Score: YamNet predictions on what sound it is. Spectrogram: Audio represented as spectrogram
    _, embeddings, _ = yamnet(waveform)

    for emb in embeddings.numpy():
        X.append(emb)
        y.append(1) #1 for a siren

# Non-sirens
for wav_file in Path("../dataset/notSirens").glob("*.wav"):
    waveform, sr = librosa.load(
        wav_file,
        sr=16000,
        mono=True
    )

    _, embeddings, _ = yamnet(waveform)

    for emb in embeddings.numpy():
        X.append(emb)
        y.append(0) #0 for non-siren

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Sirens:", np.sum(y == 1))
print("Not sirens:", np.sum(y == 0))

np.save("../data/X.npy", X)
np.save("../data/y.npy", y)
