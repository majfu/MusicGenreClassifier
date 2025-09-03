import torch
import pandas as pd


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


def print_genre_counts(labels_df):
    genre_counts = labels_df.drop(columns=['track_id']).sum().sort_values(ascending=False)
    genre_counts_df = genre_counts.reset_index()
    genre_counts_df.columns = ['genre', 'count']
    print(genre_counts_df)


def print_top_label_combinations(labels_df, top_n=10):
    genre_cols = labels_df.columns.drop('track_id')

    combos = []
    for _, row in labels_df.iterrows():
        genres = [genre for genre in genre_cols if row[genre] == 1]
        if len(genres) > 1:
            combos.append(tuple(sorted(genres)))

    combo_counts = pd.Series(combos).value_counts().reset_index()
    combo_counts.columns = ['label_combination', 'count']

    print(combo_counts.head(top_n))
