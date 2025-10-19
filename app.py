import uvicorn
import joblib
import numpy as np
import io
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ----------------------------------------------------
# 1. (เพิ่ม) โหลด TensorFlow และโมเดลสกัด Feature (MobileNetV2)
# ----------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image

# ปิดการทำงานของ GPU ถ้าไม่จำเป็น (Render ใช้ CPU)
tf.config.set_visible_devices([], 'GPU')

try:
    # ----------------------------------------------------
    # 2. โหลดโมเดลสกัด Feature (ทำนอกฟังก์ชัน)
    #    (ตามโค้ด Colab ของคุณ)
    # ----------------------------------------------------
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    print("MobileNetV2 feature extractor loaded.")

    # ----------------------------------------------------
    # 3. โหลดโมเดล SVM และ Class Indices
    # ----------------------------------------------------
    model = joblib.load("final_svm_model.joblib")
    print("SVM model loaded.")
    
    with open("class_indices.json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)
        # แปลง key ที่เป็น string '0', '1' ให้เป็น int 0, 1
        class_names = {int(k): v for k, v in class_indices.items()}
    print(f"Class names loaded: {class_names}")

except Exception as e:
    print(f"Error loading models or files: {e}")
    base_model = None
    model = None
    class_names = {}

# ----------------------------------------------------
# 4. สร้างแอป FastAPI และตั้งค่า CORS
# ----------------------------------------------------
app = FastAPI(title="Rice Classification API")

origins = ["*"] # อนุญาตทุกโดเมน

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 5. (สมบูรณ์แล้ว) ฟังก์ชัน Preprocessing
# ----------------------------------------------------
def preprocess_image(image_bytes: bytes):
    """
    ฟังก์ชันนี้ถูกแก้ไขตามโค้ด Colab ของคุณแล้ว
    1. เปิดภาพ
    2. Resize เป็น (224, 224)
    3. Rescale 1./255
    4. สกัด Feature ด้วย MobileNetV2
    """
    
    # 1. เปิดภาพจาก bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # 2. ตรวจสอบว่าภาพเป็น 3-channel (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # 3. Resize เป็น (224, 224) (ตามโค้ด Colab)
    img = img.resize((224, 224))
    
    # 4. แปลงเป็น Numpy Array
    x = keras_image.img_to_array(img)
    
    # 5. เพิ่มมิติเพื่อสร้าง "batch" (โมเดลรับ (1, 224, 224, 3))
    x = np.expand_dims(x, axis=0)
    
    # 6. (สำคัญ) Rescale 1./255 (ตามโค้ด Colab)
    x = x / 255.0
    
    # 7. สกัด Feature ด้วย MobileNetV2 ที่โหลดไว้
    features = base_model.predict(x)
    
    return features # SVM รอรับค่านี้

# ----------------------------------------------------
# 6. Endpoints (เหมือนเดิม)
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "OK", "message": "Rice Classification API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    if not model or not base_model or not class_names:
        return {"error": "Model or class names not loaded"}, 500

    try:
        # 1. อ่านไฟล์ภาพ
        image_bytes = await file.read()
        
        # 2. เตรียมข้อมูล (เรียกฟังก์ชันที่แก้ไขแล้ว)
        features = preprocess_image(image_bytes)
        
        # 3. ทำนายผล (SVM)
        prediction_index = model.predict(features)[0]
        
        # 4. หาค่าความมั่นใจ (ถ้ามี)
        if hasattr(model, "predict_proba"):
            confidence_scores = model.predict_proba(features)
            confidence = float(np.max(confidence_scores))
        else:
            confidence = 1.0 # ถ้า SVM ไม่มี predict_proba
        
        # 5. แปลง index เป็น "ชื่อ"
        predicted_class_name = class_names.get(prediction_index, "Unknown Class")
        
        # 6. ส่งผลลัพธ์กลับ
        return {
            "predicted_class": predicted_class_name,
            "confidence": confidence
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# ----------------------------------------------------
# 7. (สำหรับรันทดสอบในเครื่อง)
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)