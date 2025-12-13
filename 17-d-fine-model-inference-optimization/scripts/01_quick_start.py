import torch
from transformers.image_utils import load_image
from transformers import DFineForObjectDetection, AutoImageProcessor

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = load_image(url)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_coco")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")