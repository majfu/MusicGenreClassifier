from MGCmodel.test_multilabel_util import test_multilabel_model
from MGCmodel.test_binary_util import test_all_binary_models

# SPECTROGRAMS_PATH = "YOUR LOCAL PATH TO GOOGLE DRIVE/MGCProject/spectrograms"

if __name__ == "__main__":
    test_multilabel_model(SPECTROGRAMS_PATH)
    test_all_binary_models(SPECTROGRAMS_PATH)
