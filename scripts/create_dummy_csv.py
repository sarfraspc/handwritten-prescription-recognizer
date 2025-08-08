
import os
import pandas as pd

TRAIN_PATH = 'data/processed/Training'
VAL_PATH = 'data/processed/Validation'
TEST_PATH = 'data/processed/Testing'

def create_dummy_csv():
    train_images = os.listdir(TRAIN_PATH)
    train_labels = ["dummy_label"] * len(train_images)
    train_df = pd.DataFrame({'IMAGE': train_images, 'MEDICINE_NAME': train_labels})
    train_df.to_csv('data/processed/train.csv', index=False)

    val_images = os.listdir(VAL_PATH)
    val_labels = ["dummy_label"] * len(val_images)
    val_df = pd.DataFrame({'IMAGE': val_images, 'MEDICINE_NAME': val_labels})
    val_df.to_csv('data/processed/val.csv', index=False)

    test_images = os.listdir(TEST_PATH)
    test_labels = ["dummy_label"] * len(test_images)
    test_df = pd.DataFrame({'IMAGE': test_images, 'MEDICINE_NAME': test_labels})
    test_df.to_csv('data/processed/test.csv', index=False)

if __name__ == '__main__':
    create_dummy_csv()

