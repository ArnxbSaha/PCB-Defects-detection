import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import shutil
import random
import xml.etree.ElementTree as ET
import yaml
from ultralytics import YOLO

#Understanding the file structure of the dataset
import os
root_path = r'C:\Users\arnab\Downloads\archive\PCB_DATASET' #Primary file extracted from Kaggle

for root, dirs, files in os.walk(root_path):
   for name in dirs:
      print(os.path.join(root, name))

#Counting files for reference
def count_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    return len(files)

classifications = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

images_dir = os.path.join(root_path, 'images') #subfolder with all images
annot_dir = os.path.join(root_path, 'Annotations') #subfolder with all annotations

for subfolder in classifications:
    images_path = os.path.join(images_dir, subfolder)
    annot_path = os.path.join(annot_dir, subfolder)
    print(f'{subfolder:<15} \t\
            {count_files_in_folder(images_path)} images \t\
            {count_files_in_folder(annot_path)} annotations')


# Building function to parse dimensions for the annotations datatset
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        data.append({
            'filename': filename,
            'width': width,
            'height': height,
            'class': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })

    return data

#Using the parsing function
all_data = []

for root, dirs, files in os.walk(annot_dir):
    for name in files:
        if name.endswith('.xml'):
            xml_path = os.path.join(root, name)
            all_data.extend(parse_xml(xml_path))

annot_df = pd.DataFrame(all_data)
annot_df.head()

#To check annotations dataset-
#print(annot_df)


#Resizing the complete DATASET to a homogenous size

def resize_images(input_dir, output_dir, target_size=(640, 640)):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                resized_image = cv2.resize(image, target_size)
                # Save the resized image
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, resized_image)

resized_img_dir = os.path.join(root_path, 'images_resized')
resize_images(images_dir, resized_img_dir)

#resize annotations function
def resize_annotations(annot_df, target_size=(640, 640)):
    all_data = []

    for index, row in annot_df.iterrows():

        width_ratio = target_size[0] / row['width']
        height_ratio = target_size[1] / row['height']

        resized_xmin = int(row['xmin'] * width_ratio)
        resized_ymin = int(row['ymin'] * height_ratio)
        resized_xmax = int(row['xmax'] * width_ratio)
        resized_ymax = int(row['ymax'] * height_ratio)

        all_data.append({
            'filename': row['filename'],
            'width': target_size[0],
            'height': target_size[1],
            'class': row['class'],
            'xmin': resized_xmin,
            'ymin': resized_ymin,
            'xmax': resized_xmax,
            'ymax': resized_ymax
        })

    annot_df_resized = pd.DataFrame(all_data)
    return annot_df_resized

#resize the previously created annotated database
annot_df_resized = resize_annotations(annot_df)
annot_df_resized.head()
#print(annot_df_resized)

#Configuring the data so far to fit the YOLO format
output_path = os.path.join(root_path, 'output')
os.makedirs(output_path, exist_ok=True)

def convert_to_yolo_labels(annotation_df, classes, target_size=(640, 640)):
    yolo_labels = []

    for _, annot in annotation_df.iterrows():
        filename = annot['filename']
        width, height = annot['width'], annot['height']
        class_name = annot['class']
        xmin, ymin, xmax, ymax = annot['xmin'], annot['ymin'], annot['xmax'], annot['ymax']

        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        class_index = classes.index(class_name)

        # Append to YOLO labels list
        yolo_labels.append((filename, class_index, x_center, y_center, bbox_width, bbox_height))

    return yolo_labels

classes = ['missing_hole', 'mouse_bite', 'open_circuit',
           'short', 'spur', 'spurious_copper']
yolo_labels = convert_to_yolo_labels(annot_df_resized, classes)

#defining function for spliting dataset for training
def split_images_and_labels(images_dir, labels, output_dir, train_split=0.8, val_split=0.2):

    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/test'), exist_ok=True)

    image_labels = {}
    for label in labels:
        filename, class_index, x_center, y_center, bbox_width, bbox_height = label
        if filename not in image_labels:
            image_labels[filename] = []
        image_labels[filename].append(label)

    # Shuffle
    image_filenames = list(image_labels.keys())
    random.shuffle(image_filenames)

    num_images = len(image_filenames)
    num_train = int(num_images * train_split)
    num_val = int(num_images * val_split)

    train_filenames = image_filenames[:num_train]
    val_filenames = image_filenames[num_train:num_train + num_val]
    test_filenames = image_filenames[num_train + num_val:]

    # Write train, val, test images and labels
    for dataset, filenames in [('train', train_filenames), ('val', val_filenames), ('test', test_filenames)]:
        for filename in filenames:
          labels = image_labels[filename]
          with open(os.path.join(output_dir, f'labels/{dataset}/{os.path.splitext(filename)[0]}.txt'), 'a') as label_file:
                for label in labels:
                    _, class_index, x_center, y_center, bbox_width, bbox_height = label
                    label_file.write(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n")
            # Copy images to corresponding folders
          shutil.copy(os.path.join(images_dir, filename), os.path.join(output_dir, f'images/{dataset}/{filename}'))

split_images_and_labels(resized_img_dir, yolo_labels, output_path)

# Creating the YAML configuration file for YOLOv5
import yaml

data_yaml = {
    'train': os.path.join(output_path, 'images/train'),
    'val': os.path.join(output_path, 'images/val'),
    'nc': len(classes),
    'names': classes
}

with open('pcb_data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=False)

# Initialize the model
model = YOLO('yolov5s.pt')

# Training
model.train(data='pcb_data.yaml', epochs=50, imgsz=640)
