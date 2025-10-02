SHOULD_DELETE_MP3 = True
FMA_SUBDATASET_NAME = 'medium'
MODEL_MODE = 'combined' # can be also 'binary' or 'multilabel'

CONTENT_DRIVE_PATH = '/content/drive'
PROJECT_FOLDER_PATH = '/content/drive/MyDrive/MGCProject'

METADATA_ZIP_FILE_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata.zip'
TRACKS_METADATA_FILE_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/tracks.csv'
GENRES_METADATA_FILE_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/genres.csv'

FMA_MEDIUM_ZIP_PATH = '/content/drive/MyDrive/MGCProject/fma_medium.zip'
MP3_FILES_FOLDER_PATH = '/content/drive/MyDrive/MGCProject/fma_medium'
WAV_FILES_FOLDER_PATH = '/content/drive/MyDrive/MGCProject/wav_audio_files'

FEATURE_VECTORS_FOLDER_PATH = '/content/drive/MyDrive/MGCProject/fma_small_features'
SPECTROGRAMS_FOLDER_PATH = '/content/drive/MyDrive/MGCProject/spectrograms'

INITIAL_ENCODED_LABELS_OUTPUT_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/multilabel/labels_initial.csv'
ENCODED_LABELS_OUTPUT_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/multilabel/labels.csv'
TRAIN_SPLIT_OUTPUT_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/multilabel/train.csv'
VAL_SPLIT_OUTPUT_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/multilabel/val.csv'
TEST_SPLIT_OUTPUT_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/multilabel/test.csv'

INITIAL_ENCODED_LABELS_OUTPUT_PATH_2 = '/content/drive/MyDrive/MGCProject/fma_metadata/binary/labels_all_initial.csv'
ENCODED_LABELS_OUTPUT_PATH_2 = '/content/drive/MyDrive/MGCProject/fma_metadata/binary/labels_all.csv'
GENRES_METADATA_FOLDER_PATH = '/content/drive/MyDrive/MGCProject/fma_metadata/binary/genres_metadata'


MEAN_LOCAL_REL_PATH = 'MusicGenreClassifier/MGCmodel/multilabel/train_mean.pt'
STD_LOCAL_REL_PATH = 'MusicGenreClassifier/MGCmodel/multilabel/train_std.pt'
MULILABEL_MODEL_BEST_REL_PATH = 'MusicGenreClassifier/MGCmodel/multilabel/model.pt'

BINARY_MODELS_DIR_REL_PATH = 'MusicGenreClassifier/MGCmodel/binary/models'
GENRES_METADATA_LOCAL_DIR_PATH = 'MusicGenreClassifier/MGCmodel/binary/genres_metadata'

LOCAL_AUDIO_FILES_DIR_REL_PATH = 'MusicGenreClassifier/backend/audio_files'

# set this as your local parent path of MusicGenreClassifier
# LOCAL_ROOT_PATH = 
