
import cv2
import numpy as np

def read_image(path, img_height: int, img_width: int):
    """
    Reads and preprocesses an image file.

    Args:
        path: The path to the image file (can be a string or a tf.Tensor).
        img_height (int): The target height for resizing.
        img_width (int): The target width for resizing.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array.
    """
    try:
        # When used with tf.py_function, path is a tensor.
        if hasattr(path, 'numpy'):
            path = path.numpy().decode('utf-8')

        # Read the image in grayscale
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Handle cases where the image cannot be read
        if image is None:
            # Return a blank image if reading fails
            return (np.ones([img_height, img_width], dtype=np.float32) * 255)

        # Resize the image, maintaining aspect ratio by padding
        h, w = image.shape
        fx = w / img_width
        fy = h / img_height
        f = max(fx, fy)
        new_size = (max(min(img_width, int(w / f)), 1), max(min(img_height, int(h / f)), 1))
        
        image = cv2.resize(image, new_size)

        # Create a new image with a white background and paste the resized image onto it
        target = np.ones([img_height, img_width]) * 255
        target[0:new_size[1], 0:new_size[0]] = image
        
        # Normalize pixel values to the range [0, 1]
        image = target / 255.0
        
        return image.astype(np.float32)
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        # Return a blank image on other errors
        return (np.ones([img_height, img_width], dtype=np.float32) * 255)

