# dataset.py
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image

class FLIRThermalDataset(Dataset):
    def __init__(self, img_folder, ann_file, transform=None):
        """
        Args:
            img_folder (str): Directory with all the images.
            ann_file (str): Path to the COCO-formatted annotations file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_folder = img_folder
        try:
            with open(ann_file, 'r') as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file {ann_file} not found.")
        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        try:
            img = Image.open(img_path).convert("RGB")  # Convert to RGB for compatibility
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {img_path} not found.")

        ann_ids = [ann['id'] for ann in self.annotations['annotations'] if ann['image_id'] == img_info['id']]
        boxes = []
        labels = []
        for ann_id in ann_ids:
            ann = next(ann for ann in self.annotations['annotations'] if ann['id'] == ann_id)
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 2] * boxes[:, 3])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transform:
            img, target = self.transform(img, target)

        return img, target
