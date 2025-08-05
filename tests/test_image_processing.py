import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 

import pandas as pd
from src.preprocessing.text_processing import create_char_map
from src.preprocessing.label_encod import add_encoded_labels
from src.config import RAW_DATA_DIR

df = pd.read_csv(RAW_DATA_DIR / "Training" / "training_labels.csv")

char_map = create_char_map(df['MEDICINE_NAME'])
char_to_idx = char_map['char_to_idx']

df = add_encoded_labels(df, 'MEDICINE_NAME', char_to_idx)

print(df[['MEDICINE_NAME', 'encoded_label']].head())
