from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from typing import List
from PIL import Image
import io
import numpy as np
import cv2
from ultralytics import YOLO

#load the trained YOLO model
model = YOLO("C:/Users/arnab/Downloads/yolov5/runs/detect/train/weights/best.pt")

app = FastAPI()

#schema
class Prediction(BaseModel):
    label: str
    confidence: float
    bbox: List[int]

#prediction endpoint
@app.post("/predict", response_model=List[Prediction])
async def predict(image: UploadFile = File(...), confidence_limit: float = 0.25):
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    img_np = np.array(img)
    results = model.predict(img_np, conf=confidence_limit)
    predictions = []
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy.cpu().numpy().astype(int).flatten().tolist() 
            prediction = {
                "label": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": bbox 
            }
            predictions.append(prediction)
    
    return predictions

#visualize endpoint
@app.post("/visualize")
async def visualize(image: UploadFile = File(...), confidence_limit: float = 0.25):
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    img_np = np.array(img)
    results = model.predict(img_np, conf=confidence_limit)    
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for result in results:
        for box in result.boxes:
            bbox = box.xyxy.cpu().numpy().astype(int).flatten().tolist()
            label = result.names[int(box.cls)]
            confidence = float(box.conf)
            cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)            
            label_text = f"{label} ({confidence:.2f})"            
            font_scale = 0.5
            thickness = 1            
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)            
            cv2.rectangle(img_np, (bbox[0], bbox[1] - text_height - 10), (bbox[0] + text_width, bbox[1]), (0, 255, 0), cv2.FILLED)            
            cv2.putText(img_np, label_text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_np)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")
