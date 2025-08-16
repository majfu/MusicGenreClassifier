from src.config.config import *
from src.config.hyperparameters import MIN_GENRE_SAMPLES_COUNT
import pandas as pd
from src.utils import fma_utils
from src.utils.io_utils import load_index_mapping_df


class LabelEncoder:
    def __init__(self, tracks_metadata_file_path=None, genres_metadata_file_path=None, track_ids_to_remove=None,
                 min_genre_samples_count=None):
        self.tracks_metadata_file_path = tracks_metadata_file_path or TRACKS_METADATA_FILE_PATH
        self.genres_metadata_file_path = genres_metadata_file_path or GENRES_METADATA_FILE_PATH
        self.track_ids_to_remove = track_ids_to_remove or []
        self.min_genre_samples_count = min_genre_samples_count or MIN_GENRE_SAMPLES_COUNT

    def add_track_id_to_remove(self, track_id):
        self.track_ids_to_remove.append(track_id)

    def get_one_hot_encoded_labels_df(self):
        valid_track_genre_pairs = self.get_valid_track_genre_pairs()
        track_genre_title_pairs = self.map_genre_id_to_title(valid_track_genre_pairs)

        one_hot_encoded_genres = pd.get_dummies(track_genre_title_pairs).groupby('track_id').sum().reset_index()
        print(one_hot_encoded_genres.head())
        return self.reset_track_indices(one_hot_encoded_genres)

    def get_valid_track_genre_pairs(self):
        fma_small_metadata = self.get_small_subset_tracks_metadata_df()
        all_track_genre_pairs = self.get_all_track_genre_pairs(fma_small_metadata)
        genre_counts = self.get_genre_counts(all_track_genre_pairs)
        underrepresented_genre_ids_list = self.get_underrepresented_genre_ids_list(genre_counts)

        filtered_pairs = self.remove_entries_in_column_by_id(all_track_genre_pairs,
                                                             underrepresented_genre_ids_list,
                                                             'genre_id')
        filtered_pairs = self.remove_entries_in_column_by_id(filtered_pairs,
                                                             self.track_ids_to_remove,
                                                             'track_id')
        return filtered_pairs

    def get_small_subset_tracks_metadata_df(self):
        tracks_metadata_df = fma_utils.load(self.tracks_metadata_file_path)
        small_mask = tracks_metadata_df['set', 'subset'] <= 'small'
        return tracks_metadata_df[small_mask]

    @staticmethod
    def get_all_track_genre_pairs(tracks_metadata_df):
        track_genre_pairs = tracks_metadata_df[('track', 'genres_all')].explode().reset_index()
        track_genre_pairs.columns = ['track_id', 'genre_id']
        return track_genre_pairs

    @staticmethod
    def get_genre_counts(track_genre_pairs_df):
        genre_counts = track_genre_pairs_df['genre_id'].value_counts().reset_index()
        genre_counts.columns = ['genre_id', 'genre_count']
        return genre_counts

    def get_underrepresented_genre_ids_list(self, genre_counts_df):
        underrepresented_genre_counts = genre_counts_df[genre_counts_df['genre_count'] <= self.min_genre_samples_count]
        return underrepresented_genre_counts['genre_id'].tolist()

    @staticmethod
    def remove_entries_in_column_by_id(track_genre_pairs_df, id_list, column_name):
        return track_genre_pairs_df[~track_genre_pairs_df[column_name].isin(id_list)]

    def map_genre_id_to_title(self, track_genre_pairs_df):
        genres_metadata_df = fma_utils.load(self.genres_metadata_file_path)
        genre_titles = genres_metadata_df.reset_index()[['genre_id', 'title']]
        left_join_tables = pd.merge(track_genre_pairs_df, genre_titles, on='genre_id', how='left')
        return left_join_tables[['track_id', 'title']]

    @staticmethod
    def reset_track_indices(one_hot_encoded_genres):
        index_mapping_df = load_index_mapping_df()

        mapped_labels_df = (
            one_hot_encoded_genres
            .merge(index_mapping_df, left_on='track_id', right_on='original_track_id')
            .drop(columns=['track_id', 'original_track_id'])
            .rename(columns={'new_track_id': 'track_id'})
            .sort_values(by='track_id')
            .reset_index(drop=True)
        )
        mapped_labels_df.insert(0, 'track_id', mapped_labels_df.pop('track_id'))
        return mapped_labels_df
