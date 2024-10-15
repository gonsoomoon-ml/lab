import cv2
import numpy as np
import os
import torch_neuron

###################################################
## Load and save neuron model
###################################################

def load_neuron_model(model_path):
    model_path = os.path.abspath(model_path)
    print(f"{model_path} is given")    
    device = torch.device("cpu")  # 먼저 CPU에 로드
    model = torch.jit.load(model_path, map_location=device)
    
    return torch_neuron.DataParallel(model)

# Save 
def save_neuron_model(model, path):
    torch.jit.save(model, "../model/traced_yolo8_model_neuron.pt")


###################################################
## Preprocess image
###################################################

def preprocess_image(image_path, input_size=(640, 640)):
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 원본 이미지 크기 저장
    original_size = image.shape[:2]
    
    # 이미지 리사이즈 및 패딩
    # resized_image = letterbox(image, input_size, stride=32, auto=True)[0]
    resized_image = letterbox(image, 640,  stride=32, auto=True)[0]
    
    # BGR에서 RGB로 변환
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # 이미지를 float32로 변환하고 정규화 (0-255 -> 0-1)
    resized_image = resized_image.astype(np.float32) / 255.0
    
    # 배치 차원 추가
    resized_image = np.expand_dims(resized_image, axis=0)
    
    # 채널 순서 변경 (HWC -> CHW)
    resized_image = np.transpose(resized_image, (0, 3, 1, 2))
    
    return resized_image, original_size


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # 항상 640x640으로 설정
    new_shape = (640, 640)

    # 현재 이미지 shape
    shape = im.shape[:2]  # [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute new unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # Divide padding into 2 sides
    dw /= 2
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Ensure the final size is exactly 640x640
    im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)

    return im, r, (dw, dh)


              
###################################################
## Post-Process
###################################################

import cv2.dnn
import numpy as np

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml("../config/coco8.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def post_process_ultralytics(input_image, outputs):
    
    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640
    
    
    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # print("## detections: ", detections)
    cv2.imwrite("result_image_with_boxes.jpg", original_image)    
    
    return detections

###################################################
# Benchmarking
###################################################

import time
import torch
from ultralytics import YOLO
import numpy as np

def benchmark_inference(model, image_path, num_runs=50, num_warmup=10):
    # 워밍업 실행
    for _ in range(num_warmup):
        _ = model.predict(image_path, 
                          save=False,
                          save_txt=False, 
                          save_crop=False, 
                          save_conf=False,
                          device=[0,1,2,3]
                          )
    
    # 벤치마킹 실행
    inference_times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        results = model.predict(image_path, 
                                save=False,
                                save_txt=False, 
                                save_crop=False, 
                                save_conf=False,
                                device=['nc:0', 'nc:1', 'nc:2']
                                )
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # ms로 변환
        inference_times.append(inference_time)
    
    # 결과 계산
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    return {
        "average_time": avg_time,
        "std_dev": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "all_times": inference_times
    }




# import cv2
# import numpy as np


# # 클래스 이름 (예시)
# from utils.local_util import draw_boxes

# boxes = [
#     [224.0, 374.0, 94.5, 268.0],
#     [225.0, 374.0, 95.0, 268.0]
# ]

# scores = [0.8909, 0.8833]
# class_ids = [76, 4]
# # class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train']
# class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# image_path = "bus.jpg"
# image = cv2.imread(image_path)

# # 바운딩 박스 그리기
# result_image = draw_boxes(image, boxes, scores, class_ids, class_names)
# cv2.imwrite("result_image_with_boxes.jpg", result_image)


# def draw_boxes(image, boxes, scores, class_ids, class_names):
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         x1, y1, x2, y2 = map(int, box)
#         label = f"{class_names[class_id]}: {score:.2f}"
        
#         # 바운딩 박스 그리기
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
#         # 라벨 그리기
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.5
#         font_thickness = 1
#         label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
#         # 라벨 배경 그리기
#         cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        
#         # 라벨 텍스트 그리기
#         cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
    
#     return image

# import torch 

# def get_bounding_box_custom(result_numpy):
#     '''
#     Generate bounding boxes using raw model output
#     :param result_numpy: numpy array
#     :return: None
#     '''
#     results = result_numpy
#     results = results.squeeze(0).transpose(0, 1)  # [8400, 84] 형태로 변경

#     # 바운딩 박스, 객체성, 클래스 정보 추출
#     boxes = results[:, :4]  # [8400, 4] - (x, y, w, h) 형식
#     objectness = results[:, 4]  # [8400]
#     class_probs = results[:, 5:]  # [8400, 79]

#     # 객체성 임계값 적용 (예: 0.5)
#     mask = objectness > 0.5
#     filtered_boxes = boxes[mask]
#     filtered_objectness = objectness[mask]
#     filtered_class_probs = class_probs[mask]

#     # 클래스 예측
#     class_ids = torch.argmax(filtered_class_probs, dim=1)

#     # 최종 결과
#     final_boxes = filtered_boxes
#     final_scores = filtered_objectness
#     final_class_ids = class_ids

#     # 결과 출력 (예시)
#     for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
#         print(f"Box: {box.tolist()}, Score: {score.item():.2f}, Class: {class_id.item()}")  
