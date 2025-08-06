
import tensorflow as tf
from src.preprocessing.image_preprocessing import read_image
from src.preprocessing.text_preprocessing import CharacterProcessor
from src.models.crnn_ctc_model import build_crnn_model

class Predictor:
    """
    Handles loading the model and making predictions.
    """
    def __init__(self, model_path: str, char_map_path: str, img_height: int, img_width: int):
        """
        Initializes the Predictor.

        Args:
            model_path (str): Path to the trained model weights.
            char_map_path (str): Path to the char_map.json file.
            img_height (int): Image height.
            img_width (int): Image width.
        """
        self.char_processor = CharacterProcessor(char_map_path)
        self.img_height = img_height
        self.img_width = img_width
        num_classes = len(self.char_processor.char_map) + 1

        # Build the model and load the weights
        self.model = build_crnn_model((self.img_height, self.img_width, 1), num_classes)
        self.model.load_weights(model_path)

    def predict(self, image_path: str):
        """
        Predicts the text from a single image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The predicted text.
        """
        # Preprocess the image
        image = read_image(image_path, self.img_height, self.img_width)
        image = tf.expand_dims(image, axis=0) # Add batch dimension

        # Make the prediction
        pred = self.model.predict(image)

        # Decode the prediction using CTC greedy decoder
        decoded = tf.keras.backend.ctc_decode(pred, input_length=[pred.shape[1]], greedy=True)[0][0]
        
        # Convert the decoded sequence back to text
        text = self.char_processor.decode(decoded.numpy()[0])
        
        return text

