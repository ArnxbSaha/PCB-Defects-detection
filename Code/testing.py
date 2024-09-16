
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import yaml

#loading class names from .yaml file
def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def predict_image(image_path, model_path, yaml_path):
    model = YOLO(model_path)
    class_names = load_class_names(yaml_path)
    results = model(image_path)
    predictions = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # classification labels

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box, conf, cls in zip(predictions, confidences, classes):
        xmin, ymin, xmax, ymax = map(int, box)
        class_label = class_names[int(cls)]
        confidence = f"{conf:.2f}"
        color = (255, 0, 0) 
        cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), color, 3)
        label = f"{class_label} ({confidence})"
        cv2.putText(image_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    #display the test image with results
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# test
predict_image(r'C:\Users\arnab\Downloads\archive\PCB_DATASET\output\images\test\04_short_03.jpg', 
    model_path=r'C:\Users\arnab\Downloads\yolov5\runs\detect\train\weights\best.pt', 
    yaml_path=r'C:\Users\arnab\Downloads\yolov5\pcb_data.yaml')
