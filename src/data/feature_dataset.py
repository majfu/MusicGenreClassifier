from torch.utils.data import Dataset
import pandas as pd
from src.utils.io_utils import load_feature_tensor
from torch.utils.data import DataLoader


class FeatureDataset(Dataset):
    def __init__(self, encoded_labels_csv_path, feature_tensors_dir, transform=None, target_transform=None):
        self.audio_labels = pd.read_csv(encoded_labels_csv_path)
        self.feature_tensors_dir = feature_tensors_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        feature_tensor_path = self.feature_tensors_dir / f'{index}.pt'
        feature_tensor = load_feature_tensor(feature_tensor_path)
        label = self.audio_labels.columns[self.audio_labels.iloc[index] == 1]

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        if self.target_transform:
            label = self.target_transform(label)

        return feature_tensor, label

    def __get_global_mean_and_std(self, batch_size=32):
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=False)

        features_sum = 0.0
        squared_features_sum = 0.0
        mean = 0
        std = 0

        try:
            for features, _ in data_loader:
                features_sum += features.sum(dim=0)
                squared_features_sum += (features ** 2).sum(dim=0)

            mean = features_sum / self.__len__()
            std = (squared_features_sum / self.__len__() - mean ** 2).sqrt()

        except Exception as e:
            print("Error computing global mean and standard deviation: ", e)

        return mean, std
