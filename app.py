import uvicorn
import joblib
import numpy as np
import io
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ----------------------------------------------------
# (เพิ่มเข้ามา) ส่วนสำหรับดาวน์โหลดโมเดล
# ----------------------------------------------------
import os
import requests # เราจะใช้ library นี้

# 👇 วาง "ลิงก์ดาวน์โหลดตรง" ของคุณที่นี่
MODEL_URL = "https://drive.google.com/uc?export=download&id=SOME_LONG_ID" 
MODEL_PATH = "final_svm_model.joblib"

if not os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_PATH}' not found. Downloading from URL...")
    try:
        r = requests.get(MODEL_URL, allow_redirects=True)
        r.raise_for_status() # เช็กว่าลิงก์ไม่เสีย
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
# ----------------------------------------------------

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image

tf.config.set_visible_devices([], 'GPU')

try:
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    print("MobileNetV2 feature extractor loaded.")

    # ----------------------------------------------------
    # 3. โหลดโมเดล SVM และ Class Indices
    # (ตอนนี้มันจะโหลดไฟล์ที่เพิ่งดาวน์โหลดมา)
    # ----------------------------------------------------
    model = joblib.load(MODEL_PATH) # 👈 โหลดจากตัวแปร MODEL_PATH
    print("SVM model loaded.")
    
    with open("class_indices.json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)
        class_names = {int(k): v for k, v in class_indices.items()}
    print(f"Class names loaded: {class_names}")

except Exception as e:
    print(f"Error loading models or files: {e}")
    base_model = None
    model = None
    class_names = {}

# (โค้ดส่วนที่เหลือของ app.py ก็เหมือนเดิมครับ)
# ----------------------------------------------------
# 4. สร้างแอป FastAPI และตั้งค่า CORS
# ----------------------------------------------------
app = FastAPI(title="Rice Classification API")
# ... (โค้ดส่วนที่เหลือทั้งหมด) ...
