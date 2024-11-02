import torch
import numpy as np
from PIL import Image  # PIL 이미지 처리 라이브러리 임포트

# prepare control image
def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image


def tensor_to_image_alternative(tensor):
    """
    다른 값 범위를 가진 텐서를 위한 대체 버전
    """
    # 텐서를 CPU로 이동하고 넘파이 배열로 변환
    image = tensor.cpu().squeeze(0).numpy()
    
    # [3, 512, 512] -> [512, 512, 3] 형태로 변환
    image = np.transpose(image, (1, 2, 0))
    
    # 값 범위를 [0, 1]로 정규화
    image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype(np.uint8)
    
    return Image.fromarray(image)



def make_image_grid(images, rows, cols):
    """
    이미지들을 격자 형태로 배열하는 함수
    
    매개변수:
        images: PIL 이미지들의 리스트
        rows: 격자의 행 수
        cols: 격자의 열 수
    
    반환값:
        격자 형태로 배열된 하나의 PIL 이미지
    """
    # 모든 입력 이미지를 PIL Image 형식으로 변환
    images = [img if isinstance(img, Image.Image) else Image.fromarray(img) for img in images]
    
    # 첫 번째 이미지의 크기를 기준으로 설정
    w, h = images[0].size
    
    # 전체 격자 크기의 새 이미지 생성
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    # 각 이미지를 격자 위치에 붙여넣기
    for idx, img in enumerate(images):
        grid.paste(img, box=(idx % cols * w, idx // cols * h))
    
    return grid


################################3    
# 얼굴 사진 (고호 스타일)
################################3    

import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from PIL import Image
import numpy as np

def create_styled_portrait(
    face_image,           # 원본 얼굴 이미지
    mask_image,          # 얼굴 영역 마스크
    style_prompt,        # 스타일 프롬프트
    seed=42              # 재현성을 위한 시드
):
    # ControlNet 모델 로드
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float16
    )

    # 파이프라인 설정
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 안정적인 결과를 위한 시드 설정
    torch.manual_seed(seed)
    
    # 컨트롤 이미지 생성
    def make_inpaint_condition(init_image, mask_image):
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
        
        init_image[mask_image > 0.5] = -1.0
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image

    # 컨트롤 이미지 준비
    control_image = make_inpaint_condition(face_image, mask_image)

    # 이미지 생성
    output = pipe(
        prompt=style_prompt,
        image=face_image,
        mask_image=mask_image,
        control_image=control_image,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    return output


################################3    
# 얼굴 스와핑 인페인팅 (시도1)
################################3    

from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from PIL import Image, ImageDraw  # ImageDraw 추가

def create_simple_face_mask(image_path, center_y_ratio=0.35, mask_size_ratio=0.2):
    """
    얼굴 부분에 대한 마스크 생성
    """
    # 이미지 로드 및 리사이즈
    image = Image.open(image_path) if isinstance(image_path, str) else image_path
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 마스크 생성
    mask = Image.new('L', (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    
    # 중앙에 타원 그리기 (얼굴 위치에 맞게 조정)
    center_x = 256
    center_y = int(512 * center_y_ratio)  # 얼굴 위치 조정
    radius_x = int(256 * mask_size_ratio)
    radius_y = int(256 * mask_size_ratio * 1.3)  # 세로로 좀 더 긴 타원
    
    draw.ellipse([
        center_x - radius_x, center_y - radius_y,
        center_x + radius_x, center_y + radius_y
    ], fill=255)
    
    return image, mask

def transform_face(
    target_image_path,    # 변환할 이미지 (폭포 배경의 셀카)
    style_prompt,         # 스타일 프롬프트
    mask_size_ratio=0.2,  # 마스크 크기 비율
    center_y_ratio=0.35   # 마스크 중심 y위치 비율
):
    # ControlNet 모델 로드
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float16
    )
    
    # 파이프라인 설정
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 마스크 생성
    image, mask = create_simple_face_mask(
        target_image_path,
        center_y_ratio=center_y_ratio,
        mask_size_ratio=mask_size_ratio
    )
    
    # 컨트롤 이미지 준비
    def make_inpaint_condition(init_image, mask_image):
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
        
        init_image[mask_image > 0.5] = -1.0
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image
    
    control_image = make_inpaint_condition(image, mask)
    
    # 이미지 생성
    output = pipe(
        prompt=style_prompt,
        image=image,
        mask_image=mask,
        control_image=control_image,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    return output

################################3    
# 얼굴 스와핑 인페인팅 (시도2)
################################3    
# from PIL import Image, ImageDraw
# import numpy as np
# import torch
# from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

# def create_face_mask(image_path, center_y_ratio=0.25, mask_size_ratio=0.15):
#     """
#     화가 이미지의 얼굴 부분에 대한 마스크 생성
#     """
#     image = Image.open(image_path) if isinstance(image_path, str) else image_path
#     image = image.resize((512, 768), Image.Resampling.LANCZOS)  # 세로로 긴 비율 유지
    
#     # 마스크 생성
#     mask = Image.new('L', (512, 768), 0)
#     draw = ImageDraw.Draw(mask)
    
#     # 얼굴 위치에 타원 그리기
#     center_x = 256
#     center_y = int(768 * center_y_ratio)
#     radius_x = int(256 * mask_size_ratio)
#     radius_y = int(256 * mask_size_ratio * 1.3)
    
#     draw.ellipse([
#         center_x - radius_x, center_y - radius_y,
#         center_x + radius_x, center_y + radius_y
#     ], fill=255)
    
#     return image, mask

def replace_face(
    painter_image_path,   # 화가 이미지
    my_face_image_path,   # 내 얼굴 이미지
    mask_size_ratio=0.15, # 마스크 크기
    center_y_ratio=0.25   # 마스크 위치 (위쪽으로)
):
    # ControlNet 모델 로드
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float16
    )
    
    # 파이프라인 설정
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 화가 이미지와 마스크 생성
    image, mask = create_face_mask(
        painter_image_path,
        center_y_ratio=center_y_ratio,
        mask_size_ratio=mask_size_ratio
    )
    
    # 내 얼굴 특징을 설명하는 프롬프트 생성
    face_prompt = """an elegant man in renaissance costume, Asian face features, 
                    detailed face with realistic features, maintaining the style and atmosphere 
                    of the original renaissance painting, masterpiece quality"""
    
    # 컨트롤 이미지 준비
    def make_inpaint_condition(init_image, mask_image):
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
        
        init_image[mask_image > 0.5] = -1.0
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image
    
    control_image = make_inpaint_condition(image, mask)
    
    # 이미지 생성
    output = pipe(
        prompt=face_prompt,
        image=image,
        mask_image=mask,
        control_image=control_image,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    return output

from PIL import Image, ImageDraw
import numpy as np
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

def create_face_mask(image_path, center_y_ratio=0.25, mask_size_ratio=0.15):
    """
    화가 이미지의 얼굴 부분에 대한 마스크 생성
    """
    image = Image.open(image_path) if isinstance(image_path, str) else image_path
    image = image.resize((512, 768), Image.Resampling.LANCZOS)
    
    mask = Image.new('L', (512, 768), 0)
    draw = ImageDraw.Draw(mask)
    
    center_x = 256
    center_y = int(768 * center_y_ratio)
    radius_x = int(256 * mask_size_ratio)
    radius_y = int(256 * mask_size_ratio * 1.3)
    
    draw.ellipse([
        center_x - radius_x, center_y - radius_y,
        center_x + radius_x, center_y + radius_y
    ], fill=255)
    
    return image, mask

def blend_faces(
    painter_image_path,
    my_face_image_path,
    mask_size_ratio=0.15,
    center_y_ratio=0.25,
    strength=0.3  # 변환 강도 (낮을수록 원본 얼굴 특징 유지)
):
    # ControlNet 모델 로드
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float16
    )
    
    # 파이프라인 설정 (img2img 사용)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 화가 이미지와 마스크 준비
    painter_image, mask = create_face_mask(
        painter_image_path,
        center_y_ratio=center_y_ratio,
        mask_size_ratio=mask_size_ratio
    )
    
    # 내 얼굴 이미지 준비
    my_face = Image.open(my_face_image_path)
    my_face = my_face.resize((512, 768), Image.Resampling.LANCZOS)
    
    # 컨트롤 이미지 준비
    def make_inpaint_condition(init_image, mask_image):
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
        
        init_image[mask_image > 0.5] = -1.0
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image
    
    control_image = make_inpaint_condition(painter_image, mask)
    
    # 스타일 유지하면서 얼굴 특징 보존하는 프롬프트
    face_prompt = """highly detailed face of an Asian man in renaissance style portrait,
                    preserve original facial features, maintain ethnic characteristics,
                    seamless integration with renaissance painting style,
                    elegant costume, professional quality, masterpiece"""
    
    # 이미지 생성
    output = pipe(
        prompt=face_prompt,
        image=my_face,  # 내 얼굴 이미지를 시작점으로 사용
        control_image=control_image,
        controlnet_conditioning_scale=0.8,  # ControlNet 영향력 조절
        strength=strength,  # 변환 강도 (낮을수록 원본 유지)
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    return output

#############################################
## 사람 얼굴 이미지 교체 ( 시도 3)    
#############################################
