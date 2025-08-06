import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.preprocessing.image_preprocessing import read_image
from src.preprocessing.text_preprocessing import CharacterProcessor
from src.models.crnn_ctc_model import build_crnn_model
from src.training.train_utils import train_model

# --- Configuration ---
DATA_PATH = 'data/processed/Training'
CHAR_MAP_PATH = 'data/processed/char_map.json'
MODEL_PATH = 'models/crnn_model.h5'
IMG_HEIGHT = 64
IMG_WIDTH = 256
EPOCHS = 100
BATCH_SIZE = 32

def create_dataset(image_paths, labels, char_processor):
    """
    Creates a TensorFlow dataset from image paths and labels.
    """
    # Encode the labels
    encoded_labels = [char_processor.encode(label) for label in labels]
    
    # Pad the encoded labels to the same length
    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(encoded_labels, padding='post')

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, padded_labels))
    
    # Map the image loading and preprocessing function to the dataset
    def process_path(path, label):
        img = tf.py_function(read_image, [path, IMG_HEIGHT, IMG_WIDTH], tf.float32)
        img.set_shape([IMG_HEIGHT, IMG_WIDTH])
        img = tf.expand_dims(img, axis=-1)
        return img, label

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch the dataset
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    """
    Main training script.
    """
    # Load the training data
    train_df = pd.read_csv('data/processed/train.csv')
    image_paths = [os.path.join(DATA_PATH, img_name) for img_name in train_df['image']]
    labels = train_df['label'].tolist()

    # Create or load the character processor
    char_processor = CharacterProcessor()
    char_processor.save_char_map(CHAR_MAP_PATH)
    num_classes = len(char_processor.char_map) + 2 # Add 1 for padding (0) and 1 for CTC blank

    # Split the data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Create the datasets
    train_dataset = create_dataset(train_paths, train_labels, char_processor)
    val_dataset = create_dataset(val_paths, val_labels, char_processor)

    # Build the model
    model = build_crnn_model((IMG_HEIGHT, IMG_WIDTH, 1), num_classes)
    model.summary()

    # Train the model
    train_model(model, train_dataset, val_dataset, EPOCHS, MODEL_PATH)

if __name__ == '__main__':
    main()
