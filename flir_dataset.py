import json
import os
from datasets import Dataset

def load_flir_dataset(images_dir, annotations_file):
    """
    Loads the FLIR dataset in COCO format.

    Args:
        images_dir (str): Directory containing images.
        annotations_file (str): Path to the COCO-formatted JSON file with annotations.

    Returns:
        Dataset: A HuggingFace dataset containing image paths and annotations.
    """
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Annotation file {annotations_file} not found.")
    
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
        os.path.join(base_dir, 'images_thermal_train', 'coco.json')
    )
    val_dataset = load_flir_dataset(
        os.path.join(base_dir, 'images_thermal_val'),
        os.path.join(base_dir, 'images_thermal_val', 'coco.json')
    )
    
    return {
        'train': train_dataset,
        'validation': val_dataset
    }

if __name__ == "__main__":
    datasets = main()
    print("Datasets loaded successfully.")
