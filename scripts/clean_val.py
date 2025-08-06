
import os
import shutil

# --- Configuration ---
RAW_DATA_PATH = 'data/raw/Validation'
PROCESSED_DATA_PATH = 'data/processed/Validation'

def clean_images():
    """
    Cleans the raw images and saves them to the processed data folder.
    (Currently, this is a placeholder that just copies the files.)
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    for filename in os.listdir(RAW_DATA_PATH):
        if filename != 'validation_words':
            shutil.copy(os.path.join(RAW_DATA_PATH, filename), PROCESSED_DATA_PATH)

if __name__ == '__main__':
    clean_images()

