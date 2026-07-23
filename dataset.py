import soundata

dataset = soundata.initialize("urbansound8k", data_home="/home/gohallor/sound_datasets")
dataset.download()