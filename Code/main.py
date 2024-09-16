from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float, create_engine
import uvicorn
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("C:/Users/arnab/Downloads/yolov5/runs/detect/train/weights/best.pt")

app = FastAPI()

# fill in details in the format postgresql+psycopg2://username:password@localhost/databasename
DATABASE_URL = "postgresql+psycopg2://postgres:arnab@localhost/pcb_predict"

# setting up sqlalchemy to communicate database from API
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#for API
class Prediction(BaseModel):
    label: str
    confidence: float
    bbox: List[int]

# for Database
class PredictionModel(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String) 
    label = Column(String)
    confidence = Column(Float)
    bbox_xmin = Column(Integer)
    bbox_ymin = Column(Integer)
    bbox_xmax = Column(Integer)
    bbox_ymax = Column(Integer)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#prediction endpoint
@app.post("/predict", response_model=List[Prediction])
async def predict(image: UploadFile = File(...), confidence_limit: float = 0.25, db: SessionLocal = Depends(get_db)):
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    image_name = image.filename
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

            new_prediction = PredictionModel(
                image_name=image_name, 
                label=result.names[int(box.cls)],
                confidence=float(box.conf),
                bbox_xmin=bbox[0],
                bbox_ymin=bbox[1],
                bbox_xmax=bbox[2],
                bbox_ymax=bbox[3]
            )
            db.add(new_prediction)
    
    db.commit()

    return predictions


# visualize endpoint
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

