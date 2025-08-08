
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def read_image(path, img_height: int, img_width: int):
    try:
        if hasattr(path, 'numpy'):
            path = path.numpy().decode('utf-8')

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return (np.ones([img_height, img_width], dtype=np.float32) * 255)

        h = int(image.shape[0])
        w = int(image.shape[1])

        fx = w / img_width
        fy = h / img_height
        f = max(fx, fy)
        if f == 0:
            f = 1.0  

        new_size = (max(min(img_width, int(w / f)), 1), max(min(img_height, int(h / f)), 1))

        image = cv2.resize(image, new_size)

        target = np.ones([img_height, img_width]) * 255
        target[0:new_size[1], 0:new_size[0]] = image

        image = target / 255.0

        return image.astype(np.float32)
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return (np.ones([img_height, img_width], dtype=np.float32) * 255)

