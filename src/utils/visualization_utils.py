import matplotlib.pyplot as plt
import librosa
import numpy as np
from src.utils.metadata_utils import get_genre_titles_with_counts
from src.config.hyperparameters import MIN_GENRE_SAMPLES_COUNT


def plot_genre_distribution_graph():
    plt.figure(figsize=(13, 5))
    genre_counts = get_genre_titles_with_counts()
    genre_counts.plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Tracks")
    plt.title(f"Genre (>={MIN_GENRE_SAMPLES_COUNT} samples) Distribution In Small Subset")
    plt.show()


def plot_waveform_graph(audio_signal, sampling_rate):
    plt.figure(figsize=(13, 5))
    librosa.display.waveshow(audio_signal, sr=sampling_rate)
    plt.title('Waveform Graph')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def plot_two_waveform_graphs(waveform1, waveform2, sampling_rate):
    time_axis = np.arange(len(waveform1)) / sampling_rate

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].plot(time_axis, waveform1)
    axes[0].set_title("Before Hamming Window")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, time_axis[-1])
    axes[0].grid(True)

    axes[1].plot(time_axis, waveform2)
    axes[1].set_title("After Hamming Window")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_xlim(0, time_axis[-1])
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_waveform_and_fft_graphs(windowed_frames, fft_frames, sampling_rate):
    time_axis = np.arange(len(windowed_frames)) / sampling_rate

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    axes[0].plot(time_axis, windowed_frames)
    axes[0].set_title("Windowed")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, 0.025)
    axes[0].grid(True)

    fft_length = len(fft_frames)
    fft_indices = np.arange(fft_length)
    signal_length_seconds = fft_length / sampling_rate
    frequency_hertz = fft_indices / signal_length_seconds

    axes[1].stem(frequency_hertz, np.abs(fft_frames), linefmt='b', markerfmt=' ', basefmt='-b')
    axes[1].set_title("FFT")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_xlim(0, 5000)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_periodogram(frequency_bins, periodogram_estimates):
    plt.semilogy(frequency_bins, periodogram_estimates.T)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density")
    plt.title("Periodogram")
    plt.grid(True)
    plt.show()


def visualize_windowing_result(original_frames, windowed_frames, sampling_rate):
    frame_before_windowing = original_frames[500]
    frame_after_windowing = windowed_frames[500]

    plot_two_waveform_graphs(frame_before_windowing, frame_after_windowing, sampling_rate)


def visualize_fft_result(windowed_frames, fft_frames, sampling_rate):
    windowed_frame = windowed_frames[500]
    fft_frame = fft_frames[500]

    plot_waveform_and_fft_graphs(windowed_frame, fft_frame, sampling_rate)


def visualize_periodogram_result(frequency_bins, periodogram_estimates):
    plot_periodogram(frequency_bins, periodogram_estimates[500])


def print_feature_array_result(feature_array):
    print(f"Feature array shape is {feature_array.shape}")
    print(f"One feature vector looks like this: {feature_array[500]}")
