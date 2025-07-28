import numpy as np
import librosa
from scipy.fftpack import dct
import scipy.fft
from src.utils.audio_utils import load_audio_file, convert_ms_to_samples
from config import hyperparameters

LOWEST_FREQUENCY = 0


class FeatureExtractor:
    def __init__(self, frame_length_ms=None, hop_length_ms=None, mel_bands_number=None,
                 number_of_retained_coefficients=None, delta_window_width=None):
        self.frame_length_ms = frame_length_ms or hyperparameters.FRAME_LENGTH_MS
        self.hop_length_ms = hop_length_ms or hyperparameters.HOP_LENGTH_MS
        self.mel_bands_number = mel_bands_number or hyperparameters.MEL_BANDS_NUMBER
        self.number_of_retained_coefficients = number_of_retained_coefficients or hyperparameters.NUMBER_OF_RETAINED_COEFFICIENTS
        self.delta_window_width = delta_window_width or hyperparameters.DELTA_WINDOW_WIDTH

    def extract_features(self, wav_file_path):
        audio_signal, sampling_rate = load_audio_file(wav_file_path)
        frame_length_samples = self.get_frame_length_samples(sampling_rate)
        hop_length_samples = self.get_hop_length_samples(sampling_rate)

        frames = self.frame_audio(audio_signal, frame_length_samples, hop_length_samples)

        windowed_frames = self.window_frames(frames, frame_length_samples)

        fft_frames = self.apply_fft(windowed_frames)
        periodogram_estimates = self.get_periodogram_estimates(fft_frames, frame_length_samples)

        number_of_fft_coefficients = self.get_fft_coefficients_number(periodogram_estimates)
        mel_cepstral_coefficients = self.get_mel_cepstral_coefficients(sampling_rate, number_of_fft_coefficients,
                                                                       periodogram_estimates)

        truncated_coefficients = self.truncate_coefficients(mel_cepstral_coefficients)
        mfcc_deltas = self.calculate_mfcc_deltas(truncated_coefficients)
        mfcc_delta_deltas = self.calculate_mfcc_delta_deltas(truncated_coefficients)

        feature_vectors = np.concatenate((truncated_coefficients, mfcc_deltas, mfcc_delta_deltas), axis=1)
        return feature_vectors

    def get_frame_length_samples(self, sampling_rate):
        return convert_ms_to_samples(sampling_rate, self.frame_length_ms)

    def get_hop_length_samples(self, sampling_rate):
        return convert_ms_to_samples(sampling_rate, self.hop_length_ms)

    @staticmethod
    def frame_audio(audio_signal, frame_length_samples, hop_length_samples):
        return librosa.util.frame(audio_signal, frame_length=frame_length_samples, hop_length=hop_length_samples,
                                  axis=0)

    @staticmethod
    def window_frames(frames, frame_length_samples):
        window = np.hamming(frame_length_samples)
        windowed_frames = frames * window
        return windowed_frames

    @staticmethod
    def apply_fft(windowed_frames):
        return scipy.fft.rfft(windowed_frames)

    @staticmethod
    def get_fft_frequency_bins(window_length_samples, sampling_rate):
        return scipy.fft.rfftfreq(window_length_samples, 1 / sampling_rate)

    @staticmethod
    def get_periodogram_estimates(fft_frames, frame_length_samples):
        return (1 / frame_length_samples) * np.abs(fft_frames) ** 2

    @staticmethod
    def get_fft_coefficients_number(periodogram_estimates):
        return periodogram_estimates.shape[1] * 2 - 1

    def get_mel_cepstral_coefficients(self, sampling_rate, number_of_fft_coefficients, periodogram_estimates):
        mel_filter_banks = self.get_mel_filter_banks(sampling_rate, number_of_fft_coefficients)
        mel_spectograms = self.get_mel_spectograms(periodogram_estimates, mel_filter_banks)
        log_mel_coefficients = self.get_log_mel_coefficients(mel_spectograms)
        return self.apply_discrete_cosine_transform(log_mel_coefficients)

    def get_mel_filter_banks(self, sampling_rate, number_of_fft_coefficients):
        highest_frequency = self.get_highest_frequency(sampling_rate)
        mel_filter_banks = librosa.filters.mel(sr=sampling_rate,
                                               n_fft=number_of_fft_coefficients,
                                               n_mels=self.mel_bands_number,
                                               fmin=LOWEST_FREQUENCY,
                                               fmax=highest_frequency,
                                               norm=None)
        return mel_filter_banks

    @staticmethod
    def get_highest_frequency(sampling_rate):
        return sampling_rate / 2

    @staticmethod
    def get_mel_spectograms(periodogram_estimates, mel_filter_banks):
        return periodogram_estimates @ mel_filter_banks.T

    @staticmethod
    def get_log_mel_coefficients(mel_spectograms):
        return librosa.power_to_db(mel_spectograms)

    @staticmethod
    def apply_discrete_cosine_transform(log_mel_coefficients):
        return dct(log_mel_coefficients)

    def truncate_coefficients(self, mfcc_array):
        return mfcc_array[:, :self.number_of_retained_coefficients]

    def calculate_mfcc_deltas(self, mfcc_array):
        return librosa.feature.delta(mfcc_array, width=self.delta_window_width)

    def calculate_mfcc_delta_deltas(self, mfcc_array):
        return librosa.feature.delta(mfcc_array, order=2, width=self.delta_window_width)
