import torch
from src.utils.io_utils import load_encoded_labels_df
from src.config.config import TRAIN_SPLIT_OUTPUT_PATH


def get_genre_titles_with_counts(track_id_genre_id_pairs, genres_metadata_df):
    counts = get_genre_ids_counts(track_id_genre_id_pairs)
    genre_names = genres_metadata_df['title']
    return counts.rename(index=genre_names)


def get_genre_ids_counts(track_id_genre_id_pairs):
    return track_id_genre_id_pairs['genre_id'].value_counts()


def get_class_number(labels_df):
    return len(labels_df.columns) - 1


def get_samples_number(labels_df):
    return len(labels_df) - 1


def get_class_counts(labels_df):
    return labels_df.drop(columns=['track_id']).sum(axis=0)


def get_class_weights_tensor():
    labels_df = load_encoded_labels_df(encoded_labels_file_path=TRAIN_SPLIT_OUTPUT_PATH)
    class_number = get_class_number(labels_df)
    samples_number = get_samples_number(labels_df)

    class_counts = get_class_counts(labels_df)
    class_frequencies = class_counts / samples_number
    class_weights = 1 / class_frequencies
    normalized_class_weights = class_weights / class_weights.sum() * class_number

    return torch.from_numpy(normalized_class_weights.to_numpy())
