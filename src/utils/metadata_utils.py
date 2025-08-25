import torch
from src.utils.io_utils import load_encoded_labels_df
from src.config.config import TRAIN_SPLIT_OUTPUT_PATH


def get_genre_titles_with_counts(track_id_genre_id_pairs, genres_metadata_df):
    counts = get_genre_ids_counts(track_id_genre_id_pairs)
    genre_names = genres_metadata_df['title']
    return counts.rename(index=genre_names)


def get_genre_ids_counts(track_id_genre_id_pairs):
    return track_id_genre_id_pairs['genre_id'].value_counts()


def get_num_samples(labels_df):
    return len(labels_df)


def get_num_genres(labels_df):
    return labels_df.shape[1]


def get_class_counts(labels_df):
    return labels_df.sum(axis=0)


def convert_df_to_tensor(df):
    return torch.tensor(df.to_numpy(), dtype=torch.float)


def get_pos_weights_tensor(labels_df):
    num_samples = get_num_samples(labels_df)
    class_counts = get_class_counts(labels_df)
    pos_weights = (num_samples - class_counts) / class_counts
    return convert_df_to_tensor(pos_weights)
