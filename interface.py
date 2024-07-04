from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import Tk, filedialog

def load_image(image_path, image_type):
    """
    Load an image based on the specified type (RGB or thermal).
    
    Args:
        image_path (str): Path to the image.
        image_type (str): Type of the image, either 'rgb' or 'thermal'.
        
    Returns:
        Image: Loaded image.
    """
    image = Image.open(image_path)
    if image_type == 'rgb':
        image = image.convert("RGB")
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

def select_image():
    # Initialize tkinter
    root = Tk()
    root.withdraw()  # Hide the main window
    # Open file dialog and return the selected file path
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

def main():
    # Select image using file dialog
    image_path = select_image()
    if not image_path:
        print("No image selected.")
        return

    # Determine the image type (RGB or thermal)
    image_type = 'rgb' if 'rgb' in image_path.lower() else 'thermal'

    # Load processor and model
    processor = AutoImageProcessor.from_pretrained("./deformable_detr_flir")
    model = DeformableDetrForObjectDetection.from_pretrained("./deformable_detr_flir")

    # Load and preprocess image
    image = load_image(image_path, image_type)
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
