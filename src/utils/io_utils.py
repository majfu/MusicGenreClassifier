from src.config import config

def create_splits_files(train_df, val_df, test_df):

    train_df.to_csv(config.TRAIN_SPLIT_OUTPUT_PATH, index=False)
    val_df.to_csv(config.VAL_SPLIT_OUTPUT_PATH, index=False)
    test_df.to_csv(config.TEST_SPLIT_OUTPUT_PATH, index=False)