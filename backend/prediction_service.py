import os
import librosa
from src.config.hyperparameters import SAMPLING_RATE


class PredictionService():
    def __init__(self, file_path):
        self.file_path = file_path

    # get file from path
    # slice audio signal into 30 second clips, discard the end or pad to 30 second if is shorter
    # get probabilities for each
    # get the mean of probabilities
    # get predictions
    # return list of genres

    def load_audio_file(self, sampling_rate=SAMPLING_RATE):
        audio_signal, sampling_rate = librosa.load(self.file_path, sr=sampling_rate)
