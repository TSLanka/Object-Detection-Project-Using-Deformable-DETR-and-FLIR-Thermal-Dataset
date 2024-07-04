# inference.py
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import argparse

def run_inference(image_path):
    """
    Runs inference on the given image.

    Args:
        image_path (str): Path to the image file.
    """
    # Load the image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file {image_path} not found.")

    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained("./deformable_detr_flir")
    model = DeformableDetrForObjectDetection.from_pretrained("./deformable_detr_flir")

    # Prepare the input
    inputs = processor(images=image, return_tensors="pt")

    # Run inference
    outputs = model(**inputs)

    # Post-process the results
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # Print the results
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a thermal image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()
    run_inference(args.image_path)
