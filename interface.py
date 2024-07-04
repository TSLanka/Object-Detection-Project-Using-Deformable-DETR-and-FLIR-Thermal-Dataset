# inference.py
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def plot_results(image, boxes, labels):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def main():
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained("./deformable_detr_flir")
    model = DeformableDetrForObjectDetection.from_pretrained("./deformable_detr_flir")

    # Load and preprocess image
    image_path = "path_to_your_image.jpg"
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process and visualize results
    logits = outputs.logits
    boxes = outputs.pred_boxes

    # Convert logits and boxes to final predictions
    probas = logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.5
    probas = probas[keep]
    boxes = boxes[0, keep].cpu()

    # Scale boxes to image size
    img_w, img_h = image.size
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
    boxes = boxes * scale_fct

    # Plot results
    plot_results(image, boxes, probas)

if __name__ == "__main__":
    main()
