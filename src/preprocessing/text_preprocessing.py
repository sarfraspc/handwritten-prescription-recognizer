
import json

class CharacterProcessor:
    """
    Handles character-to-index mapping and text encoding/decoding.
    """
    def __init__(self, char_map_path: str = None):
        """
        Initializes the CharacterProcessor.

        Args:
            char_map_path (str, optional): Path to a pre-existing char_map.json file. 
                                           If not provided, a new map will be created.
        """
        if char_map_path:
            with open(char_map_path, 'r') as f:
                self.char_map = json.load(f)
        else:
            # Initialize with a 1-based character set. 0 is reserved for padding.
            # Sorting ensures a consistent mapping.
            chars = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "))
            self.char_map = {c: i + 1 for i, c in enumerate(chars)}
        
        self.idx_map = {i: c for c, i in self.char_map.items()}

    def encode(self, text: str):
        """
        Encodes a string of text into a list of integer indices.

        Args:
            text (str): The text to encode.

        Returns:
            list: A list of integers representing the encoded text.
        """
        return [self.char_map.get(char, 0) for char in text] # Return 0 for unknown characters

    def decode(self, encoded_text):
        """
        Decodes a list of integer indices back into a string.

        Args:
            encoded_text (list): A list of integers to decode.

        Returns:
            str: The decoded text.
        """
        # Ignores any indices not in the map (e.g., 0 for padding, blank label)
        return "".join([self.idx_map.get(idx, '') for idx in encoded_text])

    def save_char_map(self, path: str):
        """
        Saves the character map to a JSON file.

        Args:
            path (str): The path to save the char_map.json file.
        """
        with open(path, 'w') as f:
            json.dump(self.char_map, f)

