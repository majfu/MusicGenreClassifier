import os
from src.config import config
import torch
from src.features.feature_extractor import FeatureExtractor
import pandas as pd
from src.utils.metadata_utils import get_valid_track_genre_pairs, map_genre_id_to_title


def save_feature_array(wav_path, feature_tensor):
    output_path = (wav_path.
                   replace(config.FMA_SMALL_DATASET_FOLDER_NAME, config.FEATURE_FILES_FOLDER_NAME).
                   replace('.wav', '.pt'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(feature_tensor, output_path)


def create_and_save_feature_arrays():
    feature_extractor = FeatureExtractor()
    for root, dirs, files in os.walk(config.FMA_SMALL_FOLDER_PATH):
        for file in files:
            if not file.endswith('.wav'):
                continue

            wav_path = os.path.join(root, file)
            try:
                feature_array = feature_extractor.extract_features(wav_path)
                print(feature_array)
                feature_tensor = torch.from_numpy(feature_array).float()
                save_feature_array(wav_path, feature_tensor)

            except Exception as e:
                print(f"Error extracting features from {wav_path}: {e}")


def create_labels_file():
    """
    labels file format: csv
    each row is one sample, they are indexed
    first column is audio track id
    the rest of the columns is multi-label one-hot encoding of genre titles
    """
    labels_df = get_one_hot_encoded_labels()
    labels_df.to_csv(config.ENCODED_LABELS_OUTPUT_PATH, index=False)


def get_one_hot_encoded_labels():
    valid_track_genre_pairs = get_valid_track_genre_pairs()
    track_genre_title_pairs = map_genre_id_to_title(valid_track_genre_pairs)

    one_hot_encoded_genres = pd.get_dummies(track_genre_title_pairs).groupby('track_id').sum().reset_index()
    return one_hot_encoded_genres


def load_encoded_labels_df(encoded_labels_file_path=config.ENCODED_LABELS_OUTPUT_PATH):
    return pd.read_csv(encoded_labels_file_path)