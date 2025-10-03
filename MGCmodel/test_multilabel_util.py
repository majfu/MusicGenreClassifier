from torch.utils.data import DataLoader
from MGCmodel.focal_loss import FocalLoss
from src.data.feature_dataset import FeatureDataset
from src.features.standardization import StandardizationTransform
from src.utils.io_utils import load_feature_tensor
from backend.path_utils import *
import torch
from src.config.parameters import *
from MGCmodel.multilabel.mgc_cnn_multilabel import MultilabelMusicGenreCNN
from torchmetrics.classification import *
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def test_multilabel_model(spectrograms_path):
    mean = load_feature_tensor(get_multilabel_mean_path())
    std = load_feature_tensor(get_multilabel_std_path())
    transform = StandardizationTransform(mean, std)

    num_genres = len(GENRE_NAMES)
    test_dataset = FeatureDataset(get_multilabel_test_split_path(), spectrograms_path, transform)
    model = load_multilabel_model(num_genres)
    test_dl = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=32)

    test_loss = 0.0
    test_f1_macro = MultilabelF1Score(num_labels=num_genres, average='macro').to(DEVICE)
    test_f1 = MultilabelF1Score(num_labels=num_genres, average=None).to(DEVICE)
    test_precision = MultilabelPrecision(num_labels=num_genres, average=None).to(DEVICE)
    test_recall = MultilabelRecall(num_labels=num_genres, average=None).to(DEVICE)
    test_accuracy = MultilabelAccuracy(num_labels=num_genres, average=None).to(DEVICE)

    criterion = FocalLoss(num_classes=num_genres)

    for x_batch, y_batch in tqdm(test_dl, desc="Testing multilabel", leave=True):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.float().to(DEVICE)
        x_batch = x_batch.unsqueeze(1).contiguous(memory_format=torch.channels_last)

        logits = model(x_batch)
        batch_loss = criterion(logits, y_batch)

        test_loss += batch_loss.item()
        preds = torch.sigmoid(logits)
        test_f1_macro.update(preds, y_batch)
        test_f1.update(preds, y_batch)
        test_precision.update(preds, y_batch)
        test_recall.update(preds, y_batch)
        test_accuracy.update(preds, y_batch)

    test_loss = test_loss / len(test_dl)
    f1_macro = test_f1_macro.compute().item()
    f1_scores = test_f1.compute().cpu().tolist()
    precision_scores = test_precision.compute().cpu().tolist()
    recall_scores = test_recall.compute().cpu().tolist()
    accuracy_scores = test_accuracy.compute().cpu().tolist()

    result_output_path = get_multilabel_result_output_path()

    with open(result_output_path, "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Macro F1 Score: {f1_macro:.4f}\n")
        f.write("Per-Genre Metrics:\n")
        for i, genre in enumerate(GENRE_NAMES):
            f.write(
                f"{genre}: F1={f1_scores[i]:.4f}, Precision={precision_scores[i]:.4f}, Recall={recall_scores[i]:.4f}, Accuracy={accuracy_scores[i]:.4f}\n")

    print("Test metrics saved")


def load_multilabel_model(num_genres):
    checkpoint = torch.load(get_multilabel_model_path(), map_location=DEVICE)
    state_dict = checkpoint['model_state']
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model = MultilabelMusicGenreCNN(num_classes=num_genres)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    return model.eval()
