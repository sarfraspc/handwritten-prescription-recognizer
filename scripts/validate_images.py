import os
import cv2

def validate_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            path = os.path.join(directory, filename)
            try:
                image = cv2.imread(path)
                if image is None:
                    print(f"Warning: {path} is not a valid image.")
            except Exception as e:
                print(f"Error reading {path}: {e}")

if __name__ == '__main__':
    validate_images('data/processed/Training/')
    validate_images('data/processed/Testing/')
    validate_images('data/processed/Validation/')
