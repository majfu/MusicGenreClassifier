import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from src.config import hyperparameters
import pandas as pd


class MultilabelStratifiedSplitter:
    def __init__(self, val_ratio=None, test_ratio=None):
        self.val_ratio = val_ratio or hyperparameters.VAL_RATIO
        self.test_ratio = test_ratio or hyperparameters.TEST_RATIO

    def get_dataset_splits(self, encoded_labels_df):

        track_ids = encoded_labels_df[['track_id']].values
        labels = encoded_labels_df.drop(columns=['track_id']).values

        test_val_split_ratio = self.get_test_val_split_ratio()
        train_ids, train_labels, val_test_ids, val_test_labels = iterative_train_test_split(track_ids,
                                                                                            labels,
                                                                                            test_size=test_val_split_ratio)

        test_split_ration = self.get_test_split_ratio()
        val_ids, val_labels, test_ids, test_labels = iterative_train_test_split(val_test_ids,
                                                                                val_test_labels,
                                                                                test_size=test_split_ration)

        columns = encoded_labels_df.columns

        train_df = pd.DataFrame(np.hstack([train_ids, train_labels]), columns=columns)
        val_df = pd.DataFrame(np.hstack([val_ids, val_labels]), columns=columns)
        test_df = pd.DataFrame(np.hstack([test_ids, test_labels]), columns=columns)

        return train_df, val_df, test_df

    def get_test_val_split_ratio(self):
        return self.val_ratio + self.test_ratio

    def get_test_split_ratio(self):
        return self.val_ratio / self.get_test_val_split_ratio()
