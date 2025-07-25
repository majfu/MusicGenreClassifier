import librosa
import os
from src.config import config
from pydub import AudioSegment

SECOND_IN_MS = 1000


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


def load_audio_file(wav_file_path):
    audio_signal, sampling_rate = librosa.load(wav_file_path, sr=None)
    return audio_signal, sampling_rate


def convert_ms_to_samples(sampling_rate, length_ms):
    return int(sampling_rate * length_ms / SECOND_IN_MS)