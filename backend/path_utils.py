from pathlib import Path
from src.config.config import *


def get_binary_mean_path(genre_name):
    return Path(LOCAL_ROOT_PATH) / Path(GENRES_METADATA_LOCAL_DIR_PATH) / Path(f'{genre_name}/train_mean.pt')


def get_binary_std_path(genre_name):
    return Path(LOCAL_ROOT_PATH) / Path(GENRES_METADATA_LOCAL_DIR_PATH) / Path(f'{genre_name}/train_std.pt')


def get_binary_model_path(genre_name):
    return Path(LOCAL_ROOT_PATH) / Path(BINARY_MODELS_DIR_REL_PATH) / Path(f'model_{genre_name}.pt')


def get_multilabel_mean_path():
    return Path(LOCAL_ROOT_PATH) / Path(MEAN_LOCAL_REL_PATH)


def get_multilabel_std_path():
    return Path(LOCAL_ROOT_PATH) / Path(STD_LOCAL_REL_PATH)


def get_multilabel_model_path():
    return Path(LOCAL_ROOT_PATH) / Path(MULILABEL_MODEL_BEST_REL_PATH)


def get_audio_files_dir_path():
    return Path(LOCAL_ROOT_PATH) / Path(LOCAL_AUDIO_FILES_DIR_REL_PATH)