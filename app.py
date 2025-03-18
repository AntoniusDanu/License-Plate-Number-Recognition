from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO
import uvicorn
import os
import shutil
from util import preprocess_image, correct_plate, filter_plate_text, draw_boxes

# Inisialisasi FastAPI
app = FastAPI()

# Load model YOLO dan PaddleOCR
MODEL_PATH = "yolov8_model/best.pt"
model = YOLO(MODEL_PATH)
ocr = PaddleOCR(lang="en")

@app.post("/predict/")
async def predict_plate(file: UploadFile = File(...)):
    # Simpan file sementara
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Baca gambar
    image = cv2.imread(temp_path)
    if image is None:
        return {"error": "Invalid image file"}

    # Deteksi plat nomor dengan YOLO
    results = model(temp_path)
    detected_plates = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = image[y1:y2, x1:x2]
            
            # Simpan gambar cropped untuk OCR
            cropped_path = "cropped_plate.jpg"
            cv2.imwrite(cropped_path, plate_crop)

            # Preprocessing gambar sebelum OCR
            processed_path = preprocess_image(cropped_path)

            # Jalankan OCR
            ocr_results = ocr.ocr(processed_path, cls=True)
            height, _, _ = plate_crop.shape
            filtered_texts = filter_plate_text(ocr_results, height)

            for bbox, text in filtered_texts:
                formatted_text = correct_plate(text)
                detected_plates.append(formatted_text)
    
    os.remove(temp_path)  # Hapus file sementara
    return {"detected_plates": detected_plates}

# Jalankan server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
