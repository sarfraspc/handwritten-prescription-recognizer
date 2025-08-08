
import os
import shutil

RAW_IMAGES_DIR = 'data/raw/Validation/validation_words'
RAW_LABELS_FILE = 'data/raw/Validation/validation_labels.csv'
PROCESSED_DATA_PATH = 'data/processed/Validation'


def clean_images():
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    for filename in os.listdir(RAW_IMAGES_DIR):
        shutil.copy(os.path.join(RAW_IMAGES_DIR, filename), PROCESSED_DATA_PATH)

    shutil.copy(RAW_LABELS_FILE, PROCESSED_DATA_PATH)


if __name__ == '__main__':
    clean_images()

