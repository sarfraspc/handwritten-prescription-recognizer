import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow as tf
from src.preprocessing.image_preprocessing import read_image
from src.preprocessing.text_preprocessing import CharacterProcessor
from src.models.crnn_ctc_model import build_crnn_model
from src.training.train_utils import train_model

TRAIN_DATA_PATH = 'data/processed/Training'
VAL_DATA_PATH = 'data/processed/Validation'
CHAR_MAP_PATH = 'data/processed/char_map.json'
MODEL_PATH = 'models/crnn_model.weights.h5'
IMG_HEIGHT = 64
IMG_WIDTH = 256
EPOCHS = 100
BATCH_SIZE = 16

def create_dataset(image_paths, labels, char_processor):
    encoded_labels = [char_processor.encode(label) for label in labels]

    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(encoded_labels, padding='post', value=-1)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, padded_labels))

    def process_path(path, label):
        img = tf.py_function(read_image, [path, IMG_HEIGHT, IMG_WIDTH], tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])

        return img, label

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    train_df = pd.read_csv(os.path.join('data/processed/', 'train.csv'))
    train_image_paths = [os.path.join(TRAIN_DATA_PATH, img_name) for img_name in train_df['IMAGE']]
    train_labels = train_df['MEDICINE_NAME'].astype(str).tolist()

    val_df = pd.read_csv(os.path.join('data/processed/', 'val.csv'))
    val_image_paths = [os.path.join(VAL_DATA_PATH, img_name) for img_name in val_df['IMAGE']]
    val_labels = val_df['MEDICINE_NAME'].astype(str).tolist()

    char_processor = CharacterProcessor(CHAR_MAP_PATH)   
    char_processor.save_char_map(CHAR_MAP_PATH)
    num_classes = len(char_processor.char_map) + 1

    train_dataset = create_dataset(train_image_paths, train_labels, char_processor)
    val_dataset = create_dataset(val_image_paths, val_labels, char_processor)

    model = build_crnn_model((IMG_HEIGHT, IMG_WIDTH, 1), num_classes)
    model.summary()

    train_model(model, train_dataset, val_dataset, EPOCHS, MODEL_PATH)

if __name__ == '__main__':
    main()
