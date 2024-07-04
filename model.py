# model.py
from transformers import DetrConfig, DetrForObjectDetection

def get_detr_model(num_classes):
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_classes
    model = DetrForObjectDetection(config)
    return model