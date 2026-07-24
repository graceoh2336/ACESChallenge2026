import numpy as np
from pathlib import Path
from audio_utils import load_waveform, get_embeddings

#Create datasets
X = [] #Embeddings: What the model learns from (Large nums)
y = [] #The correct answer the model should learn to predict (1 or 0)
groups = [] #which .wav file each embedding came from (needed for a fair training/ test split later)

file_id = 0

# Sirens
for wav_file in Path("../dataset/sirens").glob("*.wav"):
    waveform = load_waveform(wav_file)
    embeddings = get_embeddings(waveform)
    for emb in embeddings:
        X.append(emb)
        y.append(1) #1 for a siren
        groups.append(file_id)
    file_id += 1

# Non-sirens
for wav_file in Path("../dataset/notSirens").glob("*.wav"):
    waveform = load_waveform(wav_file)
    embeddings = get_embeddings(waveform)
    for emb in embeddings:
        X.append(emb)
        y.append(0) #0 for non-siren
        groups.append(file_id)
    file_id += 1

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sirens:", np.sum(y == 1))
print("Not sirens:", np.sum(y == 0))

np.save("../data/X.npy", X)
np.save("../data/y.npy", y)
np.save("../data/groups.npy", groups)
