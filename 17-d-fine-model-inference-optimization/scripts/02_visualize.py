import torch
from transformers.image_utils import load_image
from transformers import DFineForObjectDetection, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont

# Load image from local path
image_path = 'image/000000039769.jpg'
image = load_image(image_path)

# Load model and processor
image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_coco")

# Prepare inputs
inputs = image_processor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=[(image.height, image.width)],
    threshold=0.5
)

# Draw bounding boxes on image
draw = ImageDraw.Draw(image)

# Try to load a font, fall back to default if not available
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    font = ImageFont.load_default()

# Color palette for different classes
colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FF8000", "#8000FF", "#0080FF", "#FF0080", "#80FF00", "#00FF80"
]

for result in results:
    for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        score_val = score.item()
        label = label_id.item()
        box_coords = [round(i, 2) for i in box.tolist()]

        # Get label name
        label_name = model.config.id2label[label]

        # Select color based on label
        color = colors[label % len(colors)]

        # Draw bounding box
        x1, y1, x2, y2 = box_coords
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        text = f"{label_name}: {score_val:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)

        # Draw label text
        draw.text((x1, y1), text, fill="white", font=font)

        # Print detection info
        print(f"{label_name}: {score_val:.2f} {box_coords}")

# Save output image
import os
os.makedirs('output/02_visualize_detection', exist_ok=True)
output_path = 'output/02_visualize_detection/000000039769_detection.jpg'
image.save(output_path)
print(f"\nOutput saved to: {output_path}")
