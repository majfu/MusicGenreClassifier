from torch.utils.data import DataLoader

from src.data.feature_dataset import FeatureDataset
from src.features.standardization import StandardizationTransform
from src.utils.io_utils import load_feature_tensor
from backend.path_utils import *
import torch
from src.config.parameters import *
from MGCmodel.binary.mgc_cnn_binary import BinaryMusicGenreCNN
from MGCmodel.binary.mgc_cnn_binary_v2 import Binary2MusicGenreCNN
from torchmetrics.classification import *
from torch import nn
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def test_all_binary_models(spectrograms_path):
    result_output_path = get_binary_result_output_path()

    for genre_name in GENRE_NAMES:
        test_split_path = get_binary_test_split_path(genre_name)
        test_results = test_binary_model(test_split_path, spectrograms_path, genre_name)
        with open(result_output_path, 'a') as f:
            f.write(
                f"{genre_name}: F1={test_results["f1"]:.4f}, Precision={test_results["precision"]:.4f}, Recall={test_results["recall"]:.4f}, Accuracy={test_results["accuracy"]:.4f}, Loss={test_results["loss"]:.4f}\n")

    print("Test metrics saved")


def test_binary_model(test_split_path, spectrograms_path, genre_name):
    mean = load_feature_tensor(get_binary_mean_path(genre_name))
    std = load_feature_tensor(get_binary_std_path(genre_name))
    transform = StandardizationTransform(mean, std)

    test_dataset = FeatureDataset(test_split_path, spectrograms_path, transform)
    model = load_binary_model(genre_name)
    test_dl = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=32)

    test_loss = 0.0
    test_accuracy = BinaryAccuracy().to(DEVICE)
    test_precision = BinaryPrecision().to(DEVICE)
    test_recall = BinaryRecall().to(DEVICE)
    test_f1 = BinaryF1Score().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    for x_batch, y_batch in tqdm(test_dl, desc="Testing binary", leave=True):
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.float().to(DEVICE, non_blocking=True)
        x_batch = x_batch.unsqueeze(1).contiguous(memory_format=torch.channels_last)

        logits = model(x_batch)
        batch_loss = criterion(logits, y_batch)

        test_loss += batch_loss.item()
        preds = torch.sigmoid(logits)
        test_accuracy.update(preds, y_batch)
        test_precision.update(preds, y_batch)
        test_recall.update(preds, y_batch)
        test_f1.update(preds, y_batch)

    loss = test_loss / len(test_dl)
    accuracy = test_accuracy.compute().item()
    precision = test_precision.compute().item()
    recall = test_recall.compute().item()
    f1_score = test_f1.compute().item()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }


def load_binary_model(genre_name):
    checkpoint = torch.load(get_binary_model_path(genre_name), map_location=DEVICE)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    if genre_name == 'Pop':
        model = Binary2MusicGenreCNN()
    else:
        model = BinaryMusicGenreCNN()
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    return model.eval()
