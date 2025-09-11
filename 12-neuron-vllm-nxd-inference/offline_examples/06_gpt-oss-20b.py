import os
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download, login

# === 설정 ===
MODEL_NAME = "openai/gpt-oss-20b"
MODEL_DIR = os.path.abspath("local-models")
ARTIFACTS_PATH = f"{MODEL_DIR}/openai/gpt-oss-20b/"

# 기존 다운로드된 모델 경로 (백업 옵션)
EXISTING_MODEL_PATH = "./gpt-oss-20b-model"

print(f"모델 이름: {MODEL_NAME}")
print(f"모델 디렉토리: {MODEL_DIR}")
print(f"아티팩트 경로: {ARTIFACTS_PATH}")
print(f"기존 모델 경로: {EXISTING_MODEL_PATH}")

# 캐시 디렉토리 설정
model_cache_name = MODEL_NAME.replace("/", "_")
cache_dir = f"./my-neuron-cache/{model_cache_name}"

print(f"캐시 디렉토리: {cache_dir}")

# 캐시 디렉토리 생성
if not os.path.exists(cache_dir):
    print(f"[안내] 캐시 디렉토리 {cache_dir}를 생성합니다...")
    os.makedirs(cache_dir, exist_ok=True)
else:
    print(f"[안내] 캐시 디렉토리가 이미 존재합니다: {cache_dir}")

# GPT-OSS-20B 모델에 최적화된 설정
tensor_parallel_size = 2
max_model_len = 2048
block_size = 2048

# 환경 변수 설정
os.environ['NEURON_COMPILE_CACHE_URL'] = cache_dir
os.environ['NEURON_COMPILED_ARTIFACTS_PATH'] = ARTIFACTS_PATH
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "2048"
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "2048"
os.environ['NEURON_CC_FLAGS'] = f"--cache_dir={cache_dir}"

print("환경 변수 설정 완료")

# GPT-OSS-20B 모델용 프롬프트 템플릿
def format_prompt(instruction, input_text=""):
    if input_text:
        return f"### 지시사항:\n{instruction}\n\n### 입력:\n{input_text}\n\n### 응답:\n"
    else:
        return f"### 지시사항:\n{instruction}\n\n### 응답:\n"

# 샘플 프롬프트 (GPT-OSS-20B 모델에 최적화된 한국어 프롬프트)
prompts = [
    format_prompt("안녕하세요! 간단한 자기소개를 해주세요."),
    format_prompt("인공지능의 미래 발전 방향에 대해 설명해주세요."),
    format_prompt("서울의 주요 관광지에 대해 알려주세요."),
    format_prompt("한국의 전통 문화에 대해 설명해주세요.")
]

# GPT-OSS-20B 모델에 최적화된 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.7,  # 적당한 창의성
    top_p=0.9,        # 다양한 응답
    repetition_penalty=1.1,  # 반복 방지
    max_tokens=512    # 적절한 응답 길이
)

def check_hf_token():
    """HuggingFace 토큰 확인 및 설정"""
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        print("[경고] HuggingFace 토큰이 설정되지 않았습니다.")
        print("[안내] 다음 중 하나의 방법으로 토큰을 설정하세요:")
        print("1. 환경 변수 설정: export HF_TOKEN=your_token_here")
        print("2. 코드에서 직접 설정: os.environ['HF_TOKEN'] = 'your_token_here'")
        print("3. HuggingFace CLI 로그인: huggingface-cli login")
        return False
    
    try:
        login(token=token)
        print("[안내] HuggingFace 토큰 인증이 완료되었습니다.")
        return True
    except Exception as e:
        print(f"[오류] HuggingFace 토큰 인증 실패: {e}")
        return False

def check_and_download_model():
    """모델 파일 확인 및 다운로드"""
    global ARTIFACTS_PATH  # global 선언을 함수 시작 부분으로 이동
    
    print("[진행] 모델 파일 확인 중...")
    
    # 필요한 파일들 확인
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json"  # tokenizer.json 대신 tokenizer_config.json 사용
    ]
    
    # 먼저 기본 경로에서 확인
    missing_files = []
    for file in required_files:
        file_path = os.path.join(ARTIFACTS_PATH, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if not missing_files:
        print(f"[안내] GPT-OSS-20B 모델 파일이 이미 {ARTIFACTS_PATH}에 있습니다.")
        return True
    
    print(f"[안내] 기본 경로에 모델 파일이 없습니다. 누락된 파일: {', '.join(missing_files)}")
    
    # 기존 다운로드된 모델에서 필요한 파일들을 확인
    print(f"[진행] 기존 다운로드된 모델에서 파일 확인 중...")
    existing_files = []
    for file in required_files:
        existing_file_path = os.path.join(EXISTING_MODEL_PATH, file)
        if os.path.exists(existing_file_path):
            existing_files.append(file)
            print(f"   ✓ {file} 파일 발견")
        else:
            print(f"   ✗ {file} 파일 누락")
    
    if existing_files:
        print(f"[안내] 기존 모델에서 {len(existing_files)}개 파일을 발견했습니다.")
        print("[안내] 기존 모델을 사용하도록 경로를 변경합니다.")
        
        # 전역 변수 업데이트
        ARTIFACTS_PATH = EXISTING_MODEL_PATH
        print(f"[안내] 모델 경로를 {ARTIFACTS_PATH}로 변경했습니다.")
        
        # 다시 파일 확인
        all_files_exist = True
        for file in required_files:
            file_path = os.path.join(ARTIFACTS_PATH, file)
            if not os.path.exists(file_path):
                all_files_exist = False
                print(f"[경고] {file} 파일이 여전히 누락되어 있습니다.")
        
        if all_files_exist:
            print("[성공] 기존 모델 파일을 사용할 수 있습니다!")
            return True
        else:
            print("[경고] 기존 모델에도 일부 파일이 누락되어 있습니다.")
    
    # 토큰 확인
    if not check_hf_token():
        print("[오류] GPT-OSS-20B 모델에 접근하려면 HuggingFace 토큰이 필요합니다.")
        print("[안내] 또는 기존 다운로드된 모델 파일을 완성해야 합니다.")
        return False
    
    print("HuggingFace Hub에서 GPT-OSS-20B 모델을 다운로드합니다...")
    os.makedirs(os.path.dirname(ARTIFACTS_PATH), exist_ok=True)
    
    try:
        print(f"[진행] 모델 다운로드 중... (대용량 파일이므로 시간이 오래 걸릴 수 있습니다)")
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=ARTIFACTS_PATH,
            local_dir_use_symlinks=False,
            resume_download=True  # 중단된 다운로드 재개
        )
        
        # 다운로드 완료 확인
        all_files_exist = True
        for file in required_files:
            file_path = os.path.join(ARTIFACTS_PATH, file)
            if not os.path.exists(file_path):
                all_files_exist = False
                print(f"[경고] {file} 파일이 여전히 누락되어 있습니다.")
        
        if not all_files_exist:
            print(f"[오류] GPT-OSS-20B 모델 다운로드에 실패했습니다.")
            return False
            
        print("[성공] GPT-OSS-20B 모델 다운로드가 완료되었습니다!")
        
    except Exception as e:
        print(f"[오류] GPT-OSS-20B 모델 다운로드 실패: {e}")
        return False
    
    return True

def main():
    print(f"[안내] GPT-OSS-20B 모델 추론을 시작합니다...")
    print(f"[안내] 모델 경로: {ARTIFACTS_PATH}")
    print(f"[안내] 캐시 디렉토리: {cache_dir}")
    
    if not check_and_download_model():
        print("[오류] GPT-OSS-20B 모델 준비에 실패했습니다. 프로그램을 종료합니다.")
        return

    print(f"[안내] GPT-OSS-20B 모델을 로드합니다...")
    
    try:
        print("[진행] LLM 객체 생성 중...")
        llm = LLM(
            model=ARTIFACTS_PATH,
            # device="neuron",  # vLLM 0.10.1.1에서는 지원되지 않음
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=4,
            max_model_len=max_model_len,
            block_size=block_size,
            # override_neuron_config도 제거
            quantization=None,  # 양자화 비활성화
            dtype="float32",   # float32 사용으로 데이터 타입 문제 해결
        )
        
        print("GPT-OSS-20B 모델 추론을 시작합니다...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("=" * 80)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"[응답 {i+1}]")
            print(f"프롬프트: {prompt}")
            print(f"생성된 텍스트: {generated_text}")
            print("-" * 80)
            
    except Exception as e:
        print(f"[오류] 모델 로드 또는 추론 중 오류 발생: {e}")
        print("[안내] Neuron 환경 설정을 확인하거나 모델 파일의 완전성을 점검해주세요.")
        print(f"[디버깅] 오류 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
