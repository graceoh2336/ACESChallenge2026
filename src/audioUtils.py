import librosa
import numpy as np
import tensorflow_hub as hub

SAMPLE_RATE = 16000
MIN_SAMPLES = int(0.98 * SAMPLE_RATE)  # YAMNet needs ~0.96s to make one embedding frame

#To create embeddings (a vector of numbers describing the audio). Convert the audio in to machine learning features using YamNet
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")


def load_waveform(path):
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(waveform) < MIN_SAMPLES:
        waveform = np.pad(waveform, (0, MIN_SAMPLES - len(waveform)))
    return waveform


def get_embeddings(waveform):
    _, embeddings, _ = yamnet(waveform)
    return embeddings.numpy()
