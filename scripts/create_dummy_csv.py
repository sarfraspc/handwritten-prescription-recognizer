
import os
import pandas as pd

# --- Configuration ---
TRAIN_PATH = 'data/processed/Training'
VAL_PATH = 'data/processed/Validation'
TEST_PATH = 'data/processed/Testing'

def create_dummy_csv():
    """
    Creates dummy CSV files for training, validation, and testing.
    """
    # Create dummy data for training
    train_images = os.listdir(TRAIN_PATH)
    train_labels = ["dummy_label"] * len(train_images)
    train_df = pd.DataFrame({'image': train_images, 'label': train_labels})
    train_df.to_csv('data/processed/train.csv', index=False)

    # Create dummy data for validation
    val_images = os.listdir(VAL_PATH)
    val_labels = ["dummy_label"] * len(val_images)
    val_df = pd.DataFrame({'image': val_images, 'label': val_labels})
    val_df.to_csv('data/processed/val.csv', index=False)

    # Create dummy data for testing
    test_images = os.listdir(TEST_PATH)
    test_labels = ["dummy_label"] * len(test_images)
    test_df = pd.DataFrame({'image': test_images, 'label': test_labels})
    test_df.to_csv('data/processed/test.csv', index=False)

if __name__ == '__main__':
    create_dummy_csv()

