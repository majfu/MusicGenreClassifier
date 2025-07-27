# we have 7997 audio files, with uneven genre distribution, no more than 1000 and no less than 100 samples
# we need to ensure that each split (train, val, test) has sufficient representation of each genre
# the split will be 70:15:15
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from src.config.config import *
import pandas as pd


def get_dataset_splits():
    encoded_labels_df = pd.read_csv(ENCODED_LABELS_OUTPUT_PATH)

    track_ids = encoded_labels_df[['track_id']].values
    labels = encoded_labels_df.drop(columns=['track_id']).values

    test_val_split_ratio = get_test_val_split_ratio()
    train_ids, train_labels, val_test_ids, val_test_labels = iterative_train_test_split(track_ids,
                                                                                        labels,
                                                                                        test_size=test_val_split_ratio)

    test_split_ration = get_test_split_ratio()
    val_ids, val_labels, test_ids, test_labels = iterative_train_test_split(val_test_ids,
                                                                            val_test_labels,
                                                                            test_size=test_split_ration)

    columns = encoded_labels_df.columns

    train_df = pd.DataFrame(np.hstack([train_ids, train_labels]), columns=columns)
    val_df = pd.DataFrame(np.hstack([val_ids, val_labels]), columns=columns)
    test_df = pd.DataFrame(np.hstack([test_ids, test_labels]), columns=columns)

    return train_df, val_df, test_df


def get_test_val_split_ratio():
    return VAL_RATIO + TEST_RATIO


def get_test_split_ratio():
    return TEST_RATIO / get_test_val_split_ratio()


def create_splits_files():
    train_df, val_df, test_df = get_dataset_splits()

    train_df.to_csv(TRAIN_SPLIT_OUTPUT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_OUTPUT_PATH, index=False)
