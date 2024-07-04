# inference.py
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image

def run_inference(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")

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
    run_inference("Data/images_thermal_val/your_test_image.jpg")  # Replace with an actual image path