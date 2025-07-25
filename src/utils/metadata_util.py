from src.utils import fma_utils
from src.config import config
import pandas as pd

tracks_metadata_file = fma_utils.load(config.tracks_metadata_dir)
genres_metadata_file = fma_utils.load(config.genres_metadata_dir)


def get_valid_track_genre_pairs():
    all_track_genre_pairs = get_all_track_genre_pairs(get_small_subset_tracks_metadata())
    genre_counts = get_genre_counts(all_track_genre_pairs)
    underrepresented_genre_ids_list = get_underrepresented_genre_ids_list(genre_counts)

    filtered_pairs = remove_specified_entries(all_track_genre_pairs, underrepresented_genre_ids_list, 'genre_id')
    filtered_pairs = remove_specified_entries(filtered_pairs, config.corrupted_track_ids_list, 'track_id')
    return filtered_pairs


def get_all_track_genre_pairs(tracks_metadata):
    track_genre_pairs = tracks_metadata[('track', 'genres_all')].explode().reset_index()
    track_genre_pairs.columns = ['track_id', 'genre_id']
    return track_genre_pairs


def get_small_subset_tracks_metadata():
    small_mask = tracks_metadata_file['set', 'subset'] <= 'small'
    return tracks_metadata_file[small_mask]


def get_genre_counts(track_genre_pairs_df):
    genre_counts = track_genre_pairs_df['genre_id'].value_counts().reset_index()
    genre_counts.columns = ['genre_id', 'genre_count']
    return genre_counts


def get_underrepresented_genre_ids_list(genre_counts_df):
    underrepresented_genre_counts = genre_counts_df[genre_counts_df['genre_count'] <= config.min_genre_samples_count]
    return underrepresented_genre_counts['genre_id'].tolist()


def remove_specified_entries(track_genre_pairs_df, id_list, column_name):
    return track_genre_pairs_df[~track_genre_pairs_df[column_name].isin(id_list)]


def map_genre_id_to_title(track_genre_pairs_df):
    genre_titles = genres_metadata_file.reset_index()[['genre_id', 'title']]
    left_join_tables = pd.merge(track_genre_pairs_df, genre_titles, on='genre_id', how='left')
    return left_join_tables[['track_id', 'title']]


def get_valid_genre_counts():
    track_genre_pairs = get_valid_track_genre_pairs()
    return track_genre_pairs['genre_id'].value_counts()


def get_genre_titles_with_counts():
    counts = get_valid_genre_counts()
    genre_names = genres_metadata_file['title']
    return counts.rename(index=genre_names)
