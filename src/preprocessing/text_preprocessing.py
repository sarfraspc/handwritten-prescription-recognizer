
import json

class CharacterProcessor:
    def __init__(self, char_map_path: str = None):
        if char_map_path:
            with open(char_map_path, 'r') as f:
                self.char_map = json.load(f)
        else:
            chars = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "))
            self.char_map = {c: i for i, c in enumerate(chars)}
        
        self.idx_map = {i: c for c, i in self.char_map.items()}

    def encode(self, text: str):
        return [self.char_map[char] for char in text if char in self.char_map]
    
    def decode(self, encoded_text):
        return "".join([self.idx_map.get(idx, '') for idx in encoded_text if idx != -1])

    def save_char_map(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.char_map, f)

