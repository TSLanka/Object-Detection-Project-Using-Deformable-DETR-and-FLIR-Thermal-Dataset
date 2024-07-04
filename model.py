# model.py
from transformers import DetrConfig, DetrForObjectDetection

def get_detr_model(num_classes):
    """
    Gets a DETR model configured for the specified number of classes.

    Args:
        num_classes (int): Number of classes for object detection.

    Returns:
        DetrForObjectDetection: A DETR model.
    """
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_classes
    model = DetrForObjectDetection(config)
    return model
