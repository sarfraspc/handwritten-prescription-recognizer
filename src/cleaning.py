# scripts/clean_images.py
import numpy as np
import cv2
import os
from pathlib import Path

def clean_and_resize_image(img_path, output_path, target_size=(128, 512)):
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read {img_path}")
        return

    h, w = image.shape
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    image = cv2.copyMakeBorder(image, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left, borderType=cv2.BORDER_CONSTANT, value=255)

    image = cv2.fastNlMeansDenoising(image, None, h=10)
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    cv2.imwrite(str(output_path), image)

def process_images(df, output_dir, image_col='image_path', target_size=(128, 512)):
    for i, row in df.iterrows():
        input_path = Path(row[image_col])
        output_path = Path(output_dir) / input_path.name
        clean_and_resize_image(input_path, output_path, target_size)
