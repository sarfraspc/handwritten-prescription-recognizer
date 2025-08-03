import json
import os

def create_char_map(texts, save_path='../data/char_map.json'):
    unique_chars = sorted(set("".join(texts)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    char_map = {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}

    os.makedirs(os.path.dirname(save_path), exist_ok=True) 

    with open(save_path, 'w') as f:
        json.dump(char_map, f, indent=2)
    print(f"Character map saved at {save_path}")

    return char_map
