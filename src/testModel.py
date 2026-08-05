import numpy as np
import tensorflow as tf
from pathlib import Path
from audioUtils import load_waveform, get_embeddings

classifier = tf.keras.models.load_model("../models/siren_classifier.keras")

for wav_file in Path("../dataset/testData/siren").rglob("*.wav"):
    
    waveform = load_waveform(wav_file)
    embeddings = get_embeddings(waveform)

    predictions = classifier.predict(
        embeddings,
        verbose=0
    )

    mean_probability = float(np.mean(predictions))

    max_probability = float(np.max(predictions))

    print(
        f"{wav_file.name:40} "
        f"{mean_probability:.4f}"
        f"{max_probability:.4f}"
    )
