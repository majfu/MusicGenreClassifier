from src.config.config import *
from src.config.hyperparameters import *
import pandas as pd
import os
import torch
from pydub import AudioSegment
import librosa
from pathlib import Path
import shutil


def create_labels_file(labels_df, output_path):
    labels_df.to_csv(output_path, index=False)


def convert_selected_mp3_to_wav(track_ids_to_convert, mp3_folder_path=MP3_FILES_FOLDER_PATH,
                                output_folder_path=WAV_FILES_FOLDER_PATH, should_delete_mp3=SHOULD_DELETE_MP3):
    mp3_folder_path = Path(mp3_folder_path)
    output_folder_path = Path(output_folder_path)
    corrupted_track_ids = []

    os.makedirs(os.path.dirname(output_folder_path), exist_ok=True)
    for mp3_file in mp3_folder_path.rglob('*.mp3'):

        track_id = int(mp3_file.stem)
        if track_id not in track_ids_to_convert:
            continue

        wav_path = get_wav_output_path(track_id, output_folder_path)
        if wav_path.exists():
            print(f"{mp3_file} already exists")
            continue

        try:
            convert_mp3_to_wav(mp3_file, wav_path)

        except Exception as e:
            corrupted_track_ids.append(track_id)
            print(f"Error converting {mp3_file}: {e}")

        if should_delete_mp3:
            mp3_file.unlink()

    return corrupted_track_ids


def convert_mp3_to_wav(input_path, output_path):
    audio_mp3_file = AudioSegment.from_mp3(input_path)
    audio_mp3_file.export(output_path, format='wav')


def get_wav_output_path(track_id, output_folder_path):
    return output_folder_path / f"{str(track_id)}.wav"


def remove_outlier_files(label_encoder, audio_files_folder_path=WAV_FILES_FOLDER_PATH):
    audio_files_path = Path(audio_files_folder_path)

    for track_id in LENGTH_OUTLIERS_TRACK_IDS:
        label_encoder.add_track_id_to_remove(track_id)

        audio_path = audio_files_path / f"{track_id}.wav"
        audio_path.unlink()


def load_audio_file(wav_file_path, sampling_rate=SAMPLING_RATE):
    audio_signal, sampling_rate = librosa.load(wav_file_path, sr=sampling_rate)
    return pad_or_truncate(audio_signal), sampling_rate


def pad_or_truncate(audio_signal):
    return librosa.util.fix_length(audio_signal, size=AUDIO_LENGTH_SAMPLES)


def load_encoded_labels_df(encoded_labels_file_path=ENCODED_LABELS_OUTPUT_PATH):
    return pd.read_csv(encoded_labels_file_path)


def create_splits_files(train_df, val_df, test_df):
    train_df.to_csv(TRAIN_SPLIT_OUTPUT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_OUTPUT_PATH, index=False)


def create_and_save_feature_arrays(feature_extractor, audio_files_folder_path=WAV_FILES_FOLDER_PATH,
                                   output_folder_path=FEATURE_VECTORS_FOLDER_PATH):
    for root, dirs, files in os.walk(audio_files_folder_path):
        for file in files:
            if not file.endswith('.wav'):
                continue

            wav_path = os.path.join(root, file)
            try:
                feature_array = feature_extractor.extract_features(wav_path)
                feature_tensor = torch.from_numpy(feature_array).float()
                save_feature_array(wav_path, feature_tensor, audio_files_folder_path, output_folder_path)

            except Exception as e:
                print(f"Error extracting features from {wav_path}: {e}")


def create_and_save_spectrograms(feature_extractor, audio_files_folder_path=WAV_FILES_FOLDER_PATH,
                                 output_folder_path=SPECTROGRAMS_FOLDER_PATH):
    for root, dirs, files in os.walk(audio_files_folder_path):
        for file in files:
            if not file.endswith('.wav'):
                continue

            wav_path = os.path.join(root, file)
            try:
                feature_array = feature_extractor.extract_spectrograms(wav_path)
                feature_tensor = torch.from_numpy(feature_array).float()
                save_feature_array(wav_path, feature_tensor, audio_files_folder_path, output_folder_path)

            except Exception as e:
                print(f"Error extracting features from {wav_path}: {e}")


def save_feature_array(wav_path, feature_tensor, audio_files_folder_path, output_folder_path):
    output_path = (wav_path.
                   replace(audio_files_folder_path, output_folder_path).
                   replace('.wav', '.pt'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(feature_tensor, output_path)


def load_feature_tensor(pt_file_path):
    return torch.load(pt_file_path).float()


def calculate_and_save_dataset_mean_and_std(dataset, mean_output_path, std_output_path):
    mean, std = dataset.get_global_mean_and_std()
    torch.save(mean, mean_output_path)
    torch.save(std, std_output_path)
