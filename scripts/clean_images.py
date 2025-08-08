
import shutil
import os

RAW_IMAGES_DIR = 'data/raw/Training/training_words'
RAW_LABELS_FILE = 'data/raw/Training/training_labels.csv'
PROCESSED_DATA_PATH = 'data/processed/Training'

if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# Copy images
for filename in os.listdir(RAW_IMAGES_DIR):
    shutil.copy(os.path.join(RAW_IMAGES_DIR, filename), PROCESSED_DATA_PATH)

# Copy labels
shutil.copy(RAW_LABELS_FILE, PROCESSED_DATA_PATH)


