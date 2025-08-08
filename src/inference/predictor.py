
import tensorflow as tf
from src.preprocessing.image_preprocessing import read_image
from src.preprocessing.text_preprocessing import CharacterProcessor
from src.models.crnn_ctc_model import build_crnn_model

class Predictor:
    def __init__(self, model_path: str, char_map_path: str, img_height: int, img_width: int):
        self.char_processor = CharacterProcessor(char_map_path)
        self.img_height = img_height
        self.img_width = img_width
        num_classes = len(self.char_processor.char_map) + 1

        self.model = build_crnn_model((self.img_height, self.img_width, 1), num_classes)
        self.model.load_weights(model_path)

    def predict(self, image_path: str):
        image = read_image(image_path, self.img_height, self.img_width)
        image = tf.expand_dims(image, axis=0)

        pred = self.model.predict(image)

        decoded = tf.keras.backend.ctc_decode(pred, input_length=[pred.shape[1]], greedy=True)[0][0]

        text = self.char_processor.decode(decoded.numpy()[0])
        
        return text

