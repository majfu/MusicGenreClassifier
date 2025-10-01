import os
from src.utils.metadata_utils import get_track_ids_list
import pandas as pd
import math
from src.config.parameters import VAL_RATIO, TEST_RATIO
from src.config.config import GENRES_METADATA_FOLDER_PATH


def create_label_files_for_each_genre(labels_all_df, genres_metadata_folder_path=GENRES_METADATA_FOLDER_PATH):
    os.makedirs(genres_metadata_folder_path, exist_ok=True)
    genre_columns = labels_all_df.drop(columns=['track_id']).columns

    for genre in genre_columns:
        genre_name = genre.replace('title_', '').replace(' ', '')

        positives_ids, negatives_ids = get_pos_neg_samples_ids(labels_all_df, genre, genre_columns)
        pos_train, pos_val, pos_test = get_train_val_test_splits(positives_ids)
        neg_train, neg_val, neg_test = get_train_val_test_splits(negatives_ids)

        train_df = get_genre_labels_df(genre_name, pos_train, neg_train)
        val_df = get_genre_labels_df(genre_name, pos_val, neg_val)
        test_df = get_genre_labels_df(genre_name, pos_test, neg_test)
        all_df = get_genre_labels_df(genre_name, positives_ids, negatives_ids)

        genre_folder_path = os.path.join(genres_metadata_folder_path, genre_name)
        os.makedirs(genre_folder_path, exist_ok=True)
        save_genre_labels('train', genre_folder_path, train_df)
        save_genre_labels('val', genre_folder_path, val_df)
        save_genre_labels('test', genre_folder_path, test_df)
        save_genre_labels('all', genre_folder_path, all_df)


def save_genre_labels(split, genre_folder_path, genre_labels_df):
    genre_labels_output_path = os.path.join(str(genre_folder_path), f'{split}.csv')
    genre_labels_df.to_csv(genre_labels_output_path, index=False)


def get_train_val_test_splits(ids, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    num_val = math.floor(len(ids) * val_ratio)
    ids_val = pd.Series(ids).sample(n=num_val, random_state=37).tolist()
    ids = [i for i in ids if i not in ids_val]

    num_test = math.floor(len(ids) * test_ratio)
    ids_test = pd.Series(ids).sample(n=num_test, random_state=37).tolist()
    ids_train = [i for i in ids if i not in ids_test]

    return ids_train, ids_val, ids_test


def get_genre_labels_df(genre_name, positives_ids, negatives_ids):
    return pd.DataFrame({
        'track_id': positives_ids + negatives_ids,
        f'{genre_name}': [1] * len(positives_ids) + [0] * len(negatives_ids)
    })


def get_pos_neg_samples_ids(labels_all_df, genre, genre_columns):
    positive_samples = labels_all_df[labels_all_df[genre] == 1]
    positives_ids = get_track_ids_list(positive_samples)
    positives_number = len(positives_ids)

    negative_samples = labels_all_df[labels_all_df[genre] == 0]
    negative_genre_columns = [g for g in genre_columns if g != genre]
    num_samples_per_genre = positives_number // len(negative_genre_columns) + 1

    negatives_ids = get_samples_from_each_negative_genre(negative_genre_columns, negative_samples,
                                                         num_samples_per_genre)
    negatives_ids = down_or_upsample_negatives(positives_number, negatives_ids, negative_samples)

    return positives_ids, negatives_ids


def get_samples_from_each_negative_genre(negative_genre_columns, negative_samples, num_samples_per_genre):
    negatives_ids = []

    for negative_genre in negative_genre_columns:
        negative_genre_samples = negative_samples[negative_samples[negative_genre] == 1]
        num_to_sample = min(num_samples_per_genre, len(negative_genre_samples))
        sampled = negative_genre_samples.sample(n=num_to_sample, random_state=37)

        negatives_ids += get_track_ids_list(sampled)
        negative_samples = negative_samples[~negative_samples['track_id'].isin(negatives_ids)]

    return negatives_ids


def down_or_upsample_negatives(positives_number, negatives_ids, negative_samples):
    difference_pos_neg = positives_number - len(negatives_ids)

    if difference_pos_neg < 0:
        negatives_ids = pd.Series(negatives_ids).sample(n=positives_number, random_state=37).tolist()

    if difference_pos_neg > 0:
        not_sampled_negative_samples = negative_samples[~negative_samples['track_id'].isin(negatives_ids)]
        sampled = not_sampled_negative_samples.sample(n=difference_pos_neg, random_state=37)
        negatives_ids += get_track_ids_list(sampled)

    return negatives_ids
