import soundata
from pathlib import Path

dataset = soundata.initialize("urbansound8k", data_home=str(Path.home() / "sound_datasets"))
dataset.download()
