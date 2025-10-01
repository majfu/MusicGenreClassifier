from src.features.feature_extractor import FeatureExtractor
from src.features.standardization import StandardizationTransform
from src.config.parameters import *
from src.config.config import *
from src.utils.io_utils import *


class PreprocessingPipeline:
    def __init__(self, audio_signal, sampling_rate):
        self.audio_signal = audio_signal
        self.sampling_rate = sampling_rate

    def extract_spectrograms(self):
        audio_slices = self.slice_audio_signal(self.audio_signal)
        feature_extractor = FeatureExtractor()
        spectrograms = [feature_extractor.get_spectrogram_from_loaded_file(audio_slice, self.sampling_rate) for
                        audio_slice in audio_slices]
        return [torch.from_numpy(spectrogram).float() for spectrogram in spectrograms]

    @staticmethod
    def standardize(spectrograms, local_mean_path, local_std_path):
        mean = load_feature_tensor(local_mean_path)
        std = load_feature_tensor(local_std_path)
        transform = StandardizationTransform(mean, std)
        return [transform(audio_slice) for audio_slice in spectrograms]

    def slice_audio_signal(self, audio_signal):
        audio_slices = [audio_signal[index:index + AUDIO_LENGTH_SAMPLES] for index in
                        range(0, len(audio_signal), AUDIO_LENGTH_SAMPLES)]

        if len(audio_slices) > 1:
            if self.is_shorter_than_threshold(audio_slices[-1]):
                del audio_slices[-1]
            else:
                audio_slices[-1] = pad_or_truncate(audio_slices[-1])
        else:
            audio_slices[0] = pad_or_truncate(audio_slices[0])

        return audio_slices

    @staticmethod
    def is_shorter_than_threshold(audio_slice):
        discard_threshold = AUDIO_LENGTH_SAMPLES / 2
        return len(audio_slice) < discard_threshold
