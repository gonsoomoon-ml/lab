import sys, os
print(os.getcwd())
sys.path.append(os.path.abspath(".."))

from utils.local_util import draw_boxes
import cv2


# 바운딩 박스 결과 (예시)
boxes = [
    [670.83, 380.08, 809.86, 879.69],
    [221.62, 407.06, 343.53, 856.26],
    [50.671, 397.60, 244.20, 905.07],
    [31.541, 230.63, 801.53, 775.84],
    [0.42298, 549.81, 57.900, 868.34]
]
scores = [0.8909, 0.8833, 0.8779, 0.8442, 0.4408]
class_ids = [0, 0, 0, 5, 0]

# 클래스 이름 (예시)
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train']

image_path = "bus.jpg"
image = cv2.imread(image_path)

# 바운딩 박스 그리기
result_image = draw_boxes(image, boxes, scores, class_ids, class_names)
cv2.imwrite("result_image_with_boxes.jpg", result_image)
# 결과 이미지 저장
# cv2.imwrite("result_image_with_boxes.jpg", result_image)

# 결과 이미지 표시 (선택사항)
