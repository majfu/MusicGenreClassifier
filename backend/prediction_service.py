import io

import librosa
from src.config.parameters import *
from backend.preprocessing_pipeline import PreprocessingPipeline
import torch
from MGCmodel.approach1.mgc_cnn_multilabel import MultilabelMusicGenreCNN
from pydub import AudioSegment
from backend.path_utils import *
from MGCmodel.approach2.mgc_cnn_binary import BinaryMusicGenreCNN
from MGCmodel.approach2.mgc_cnn_binary_v2 import Binary2MusicGenreCNN


class PredictionService:
    def __init__(self, file_path, model_mode):
        self.file_path = file_path
        self.model_mode = model_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = THRESHOLD
        self.genre_names_array = GENRE_NAMES

    def get_predictions(self):
        if self.file_path == '' or not Path(self.file_path).exists():
            return 'File does not exist', []

        try:
            audio_signal, sampling_rate = self.load_audio_file()
            preprocessing_pipeline = PreprocessingPipeline(audio_signal, sampling_rate)
            spectrograms_array = preprocessing_pipeline.extract_spectrograms()

            if self.model_mode == 'binary':
                probabilities = self.get_binary_probabilities(preprocessing_pipeline, spectrograms_array)
            elif self.model_mode == 'multilabel':
                probabilities = self.get_multilabel_probabilities(preprocessing_pipeline, spectrograms_array)
            else:
                binary_probabilities = self.get_binary_probabilities(preprocessing_pipeline, spectrograms_array)
                multilabel_probabilities = self.get_multilabel_probabilities(preprocessing_pipeline, spectrograms_array)
                probabilities = (binary_probabilities + multilabel_probabilities) / 2

            predicted_labels = (probabilities >= THRESHOLD).int().cpu().numpy()

        except Exception as e:
            return f'The was an error while classifying the genre: {e}', []

        print(predicted_labels)
        if predicted_labels.sum() == 0:
            return 'Could not classify the genre :(', []

        return 'OK', self.map_to_genre_names(predicted_labels)

    def get_multilabel_probabilities(self, preprocessing_pipeline, spectrograms_array):
        standardized_spectrograms = preprocessing_pipeline.standardize(spectrograms_array,
                                                                       get_multilabel_mean_path(),
                                                                       get_multilabel_std_path())
        model = self.load_model(False)
        probabilities = torch.zeros(len(self.genre_names_array), device=self.device)
        multilabel_probabilities = self.predict_probabilities(standardized_spectrograms, model, probabilities)

        return multilabel_probabilities

    def get_binary_probabilities(self, preprocessing_pipeline, spectrograms_array):
        binary_probabilities = torch.zeros(len(GENRE_NAMES), device=self.device)

        for index, genre in enumerate(GENRE_NAMES):
            standardized_spectrograms = preprocessing_pipeline.standardize(spectrograms_array,
                                                                           get_binary_mean_path(genre),
                                                                           get_binary_std_path(genre))
            model = self.load_model(True, genre)
            probabilities = torch.zeros(1, device=self.device)
            mean_probability = self.predict_probabilities(standardized_spectrograms, model, probabilities)
            binary_probabilities[index] = mean_probability

        return binary_probabilities

    def map_to_genre_names(self, predicted_labels):
        return [self.genre_names_array[index] for index, prediction in enumerate(predicted_labels) if prediction == 1]

    def predict_probabilities(self, standardized_spectrograms, model, probabilities):
        for spectrogram in standardized_spectrograms:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = model(spectrogram)
                probabilities += torch.sigmoid(logits).squeeze(0).squeeze(0)
        mean_probabilities = probabilities / len(standardized_spectrograms)
        return mean_probabilities

    def load_model(self, is_binary_model, genre_name=None):
        if is_binary_model:
            checkpoint = torch.load(get_binary_model_path(genre_name), map_location=self.device)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
            if genre_name == 'Pop':
                model = Binary2MusicGenreCNN()
            else:
                model = BinaryMusicGenreCNN()

        else:
            checkpoint = torch.load(get_multilabel_model_path(), map_location=self.device)
            state_dict = checkpoint['model_state']
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model = MultilabelMusicGenreCNN(num_classes=len(self.genre_names_array))

        model.load_state_dict(state_dict)
        model.to(self.device)
        return model.eval()

    def load_audio_file(self, sampling_rate=SAMPLING_RATE):
        if self.file_path.endswith('mp3'):
            audio = AudioSegment.from_mp3(self.file_path)
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            audio_signal, sampling_rate = librosa.load(wav_io, sr=sampling_rate)

        else:
            audio_signal, sampling_rate = librosa.load(self.file_path, sr=sampling_rate)

        return audio_signal, sampling_rate
