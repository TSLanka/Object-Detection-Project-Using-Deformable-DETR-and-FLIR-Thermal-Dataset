# flir_dataset.py
import json
import os
from datasets import Dataset

def load_flir_dataset(images_dir, annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    data = {
        'image_path': [os.path.join(images_dir, img['file_name']) for img in annotations['images']],
        'annotations': [
            [
                {
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id']
                }
                for ann in annotations['annotations']
                if ann['image_id'] == img['id']
            ]
            for img in annotations['images']
        ]
    }
    
    return Dataset.from_dict(data)

def main():
    base_dir = 'Data'
    train_dataset = load_flir_dataset(
        os.path.join(base_dir, 'images_thermal_train'),
        os.path.join(base_dir, 'thermal_annotations_train.json')  # You'll need to create this
    )
    val_dataset = load_flir_dataset(
        os.path.join(base_dir, 'images_thermal_val'),
        os.path.join(base_dir, 'thermal_annotations_val.json')  # You'll need to create this
    )
    
    return {
        'train': train_dataset,
        'validation': val_dataset
    }