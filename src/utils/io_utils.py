from src.config.config import *
import pandas as pd
import os
import torch
from pydub import AudioSegment
import librosa
from pathlib import Path
import shutil


def move_mp3_files_to_folder(mp3_root_folder_path=FMA_SMALL_FOLDER_PATH,
                             destination_folder_path=AUDIO_FILES_FOLDER_PATH):
    root_path = Path(mp3_root_folder_path)
    destination_path = Path(destination_folder_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    for mp3_file in root_path.rglob('*.mp3'):
        try:
            destination_file_path = destination_path / mp3_file.name
            shutil.move(str(mp3_file), str(destination_file_path))

        except Exception as e:
            print(f"Error moving {mp3_file}: {e}")


def convert_mp3_to_wav_in_directory(label_encoder, root_folder_path=AUDIO_FILES_FOLDER_PATH,
                                    should_delete_mp3=SHOULD_DELETE_MP3):
    root_path = Path(root_folder_path)

    for mp3_file in root_path.rglob('*.mp3'):
        try:
            wav_path = mp3_file.with_suffix('.wav')
            convert_mp3_to_wav(mp3_file, wav_path)

        except Exception as e:
            track_id = mp3_file.stem
            label_encoder.add_track_id_to_remove(track_id)
            print(f"Error converting {mp3_file}: {e}")

        if should_delete_mp3:
            mp3_file.unlink()


def rename_wav_files_with_reset_index(audio_files_folder_path=AUDIO_FILES_FOLDER_PATH,
                                      index_mapping_output_path=INDEX_MAPPING_FILE_PATH):
    index_mapping_df = pd.DataFrame(columns=['original_track_id', 'new_track_id'])
    audio_files_path = Path(audio_files_folder_path)
    index_mapping_path = Path(index_mapping_output_path)

    for index, wav_file in enumerate(audio_files_path.rglob('*.wav'), start=0):
        old_file_name = wav_file.stem
        new_file_name = f'{index}.wav'
        output_path = audio_files_path / new_file_name
        shutil.move(str(wav_file), str(output_path))
        index_mapping_df.loc[len(index_mapping_df)] = [old_file_name, index]

    index_mapping_df.to_csv(index_mapping_path, index=False)


def load_index_mapping_df(index_mapping_file_path=INDEX_MAPPING_FILE_PATH):
    return pd.read_csv(index_mapping_file_path)


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


def create_and_save_feature_arrays(feature_extractor, audio_files_folder_path=AUDIO_FILES_FOLDER_PATH,
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


def save_feature_array(wav_path, feature_tensor, audio_files_folder_path, output_folder_path):
    output_path = (wav_path.
                   replace(audio_files_folder_path, output_folder_path).
                   replace('.wav', '.pt'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(feature_tensor, output_path)


def load_feature_tensor(pt_file_path):
    return torch.load(pt_file_path).float()
