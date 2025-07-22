import config
import hyperparameters
import os
from pydub import AudioSegment
import librosa
import numpy as np
import scipy.fft
from scipy.fftpack import dct
import torch

SECOND_IN_MS = 1000
LOWEST_FREQUENCY = 0


# convert mp3 to wav in given directory
def convert_mp3_to_wav_in_directory(fma_small_folder_path):
    for root, dirs, files in os.walk(fma_small_folder_path):
        for file in files:
            if not file.endswith('.mp3'):
                continue
            mp3_path = os.path.join(root, file)
            wav_path = mp3_path.replace('.mp3', '.wav')

            try:
                convert_mp3_to_wav(mp3_path, wav_path)
                if config.should_delete_mp3:
                    os.remove(mp3_path)

            except Exception as e:
                print(f"Error converting {mp3_path}: {e}")


def convert_mp3_to_wav(input_path, output_path):
    audio_mp3_file = AudioSegment.from_mp3(input_path)
    audio_mp3_file.export(output_path, format='wav')


# load audio file
def load_audio_file(wav_file_path):
    audio_signal, sampling_rate = librosa.load(wav_file_path, sr=None)
    return audio_signal, sampling_rate


def convert_ms_to_samples(sampling_rate, length_ms):
    return int(sampling_rate * length_ms / SECOND_IN_MS)


def get_frame_length_samples(sampling_rate, frame_length_ms=hyperparameters.frame_length_ms):
    return convert_ms_to_samples(sampling_rate, frame_length_ms)


def get_hop_length_samples(sampling_rate, hop_length_ms=hyperparameters.hop_length_ms):
    return convert_ms_to_samples(sampling_rate, hop_length_ms)


# framing
def get_audio_frames(audio_signal, frame_length_samples, hop_length_samples):
    return librosa.util.frame(audio_signal, frame_length=frame_length_samples, hop_length=hop_length_samples, axis=0)


# windowing
def get_windowed_frames(frames, frame_length_samples):
    window = np.hamming(frame_length_samples)
    windowed_frames = frames * window
    return windowed_frames


# FFT, scipy fft is faster than numpy fft
def get_fft_frames(windowed_frames):
    return scipy.fft.rfft(windowed_frames)


def get_fft_frequency_bins(window_length_samples, sampling_rate):
    return scipy.fft.rfftfreq(window_length_samples, 1 / sampling_rate)


# periodogram estimate
def get_periodogram_estimates(fft_frames, frame_length_samples):
    return (1 / frame_length_samples) * np.abs(fft_frames) ** 2


# extracting MFCCs
def get_fft_coefficients_number(periodogram_estimates):
    return periodogram_estimates.shape[1] * 2 - 1


def get_highest_frequency(sampling_rate):
    return sampling_rate / 2


def get_mel_filter_banks(sampling_rate, number_of_fft_coefficients):
    highest_frequency = get_highest_frequency(sampling_rate)
    mel_filter_banks = librosa.filters.mel(sr=sampling_rate,
                                           n_fft=number_of_fft_coefficients,
                                           n_mels=hyperparameters.mel_bands_number,
                                           fmin=LOWEST_FREQUENCY,
                                           fmax=highest_frequency,
                                           norm=None)
    return mel_filter_banks


def get_mel_spectograms(periodogram_estimates, mel_filter_banks):
    return periodogram_estimates @ mel_filter_banks.T


def get_log_mel_coefficients(mel_spectograms):
    return librosa.power_to_db(mel_spectograms)


def apply_discrete_cosine_transform(log_mel_coefficients):
    return dct(log_mel_coefficients)


def get_mel_cepstral_coefficients(sampling_rate, number_of_fft_coefficients, periodogram_estimates):
    mel_filter_banks = get_mel_filter_banks(sampling_rate, number_of_fft_coefficients)
    mel_spectograms = get_mel_spectograms(periodogram_estimates, mel_filter_banks)
    log_mel_coefficients = get_log_mel_coefficients(mel_spectograms)
    return apply_discrete_cosine_transform(log_mel_coefficients)


def get_truncated_coefficients(mfcc_array, number_to_retain=hyperparameters.number_of_retained_coefficients):
    return mfcc_array[:, :number_to_retain]


# extracting deltas
def calculate_mfcc_deltas(mfcc_array):
    return librosa.feature.delta(mfcc_array, width=hyperparameters.delta_window_width)


# extracting delta-deltas
def calculate_mfcc_delta_deltas(mfcc_array):
    return librosa.feature.delta(mfcc_array, order=2, width=hyperparameters.delta_window_width)


# create feature vectors
def get_feature_array(wav_file_path):
    audio_signal, sampling_rate = load_audio_file(wav_file_path)
    frame_length_samples = get_frame_length_samples(sampling_rate)
    hop_length_samples = get_hop_length_samples(sampling_rate)

    frames = get_audio_frames(audio_signal, frame_length_samples, hop_length_samples)

    windowed_frames = get_windowed_frames(frames, frame_length_samples)
    
    fft_frames = get_fft_frames(windowed_frames)
    periodogram_estimates = get_periodogram_estimates(fft_frames, frame_length_samples)

    number_of_fft_coefficients = get_fft_coefficients_number(periodogram_estimates)
    mel_cepstral_coefficients = get_mel_cepstral_coefficients(sampling_rate, number_of_fft_coefficients,
                                                              periodogram_estimates)

    truncated_coefficients = get_truncated_coefficients(mel_cepstral_coefficients)
    mfcc_deltas = calculate_mfcc_deltas(truncated_coefficients)
    mfcc_delta_deltas = calculate_mfcc_delta_deltas(truncated_coefficients)

    feature_vectors = np.concatenate((truncated_coefficients, mfcc_deltas, mfcc_delta_deltas), axis=1)
    return feature_vectors


# iterate over dataset and save feature arrays
def save_feature_array(wav_path, feature_tensor):
    output_path = (wav_path.
                   replace(config.fma_small_dataset_folder_name, config.feature_files_folder_name).
                   replace('.wav', '.pt'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(feature_tensor, output_path)


def create_and_save_feature_arrays(fma_small_folder_path):
    for root, dirs, files in os.walk(fma_small_folder_path):
        for file in files:
            if not file.endswith('.wav'):
                continue

            wav_path = os.path.join(root, file)
            try:
                feature_array = get_feature_array(wav_path)
                feature_tensor = torch.from_numpy(feature_array).float()
                save_feature_array(wav_path, feature_tensor)

            except Exception as e:
                print(f"Error extracting features from {wav_path}: {e}")


# standardizing the features
