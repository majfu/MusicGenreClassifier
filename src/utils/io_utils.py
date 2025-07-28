from src.config.config import *
import pandas as pd
import os
import torch
from pydub import AudioSegment
import librosa


def convert_mp3_to_wav_in_directory(root_folder_path=FMA_SMALL_FOLDER_PATH,
                                    should_delete_mp3=SHOULD_DELETE_MP3):
    for root, dirs, files in os.walk(root_folder_path):
        for file in files:
            if not file.endswith('.mp3'):
                continue
            mp3_path = os.path.join(root, file)
            wav_path = mp3_path.replace('.mp3', '.wav')

            try:
                convert_mp3_to_wav(mp3_path, wav_path)
                if should_delete_mp3:
                    os.remove(mp3_path)

            except Exception as e:
                print(f"Error converting {mp3_path}: {e}")


def convert_mp3_to_wav(input_path, output_path):
    audio_mp3_file = AudioSegment.from_mp3(input_path)
    audio_mp3_file.export(output_path, format='wav')


def load_audio_file(wav_file_path):
    audio_signal, sampling_rate = librosa.load(wav_file_path, sr=None)
    return audio_signal, sampling_rate


def create_labels_file(labels_df):
    labels_df.to_csv(ENCODED_LABELS_OUTPUT_PATH, index=False)


def load_encoded_labels_df(encoded_labels_file_path=ENCODED_LABELS_OUTPUT_PATH):
    return pd.read_csv(encoded_labels_file_path)


def create_splits_files(train_df, val_df, test_df):
    train_df.to_csv(TRAIN_SPLIT_OUTPUT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_OUTPUT_PATH, index=False)


def create_and_save_feature_arrays(feature_extractor, root_folder=FMA_SMALL_FOLDER_PATH):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if not file.endswith('.wav'):
                continue

            wav_path = os.path.join(root, file)
            try:
                feature_array = feature_extractor.extract_features(wav_path)
                feature_tensor = torch.from_numpy(feature_array).float()
                save_feature_array(wav_path, feature_tensor)

            except Exception as e:
                print(f"Error extracting features from {wav_path}: {e}")


def save_feature_array(wav_path, feature_tensor):
    output_path = (wav_path.
                   replace(FMA_SMALL_FOLDER_PATH, FEATURE_VECTORS_FOLDER_PATH).
                   replace('.wav', '.pt'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(feature_tensor, output_path)
