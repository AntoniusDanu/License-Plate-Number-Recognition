import cv2
import numpy as np
import re

def preprocess_image(image_path):
    """Preprocessing gambar sebelum OCR."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error membaca gambar dari {image_path}")

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_image = cv2.bitwise_not(binary)

    temp_path = "processed_temp.jpg"
    cv2.imwrite(temp_path, processed_image)
    return temp_path

def correct_plate(text):
    """Koreksi kesalahan OCR pada plat nomor Indonesia."""
    corrections = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "P": "9"}
    return "".join(corrections.get(c, c) for c in text.upper())

def filter_plate_text(results, image_height):
    """Menghapus teks tidak relevan berdasarkan posisi bounding box."""
    filtered_results = []
    for line in results[0]:
        if isinstance(line, list) and len(line) >= 2:
            bbox, data = line[:2]
            if not isinstance(data, tuple) or len(data) < 2:
                continue
            text, confidence = data
            filtered_results.append((bbox, text))
    return filtered_results
