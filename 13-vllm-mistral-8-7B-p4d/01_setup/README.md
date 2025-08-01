# vLLM Mixtral-8x7B 실행 가이드

## 1. Conda 가상환경 설정

### 1.1. 가상환경 생성
```bash
# 디렉토리 이동
cd ~/lab/13-vllm-mistral-8-7B-p4d/01_setup

# 가상환경 생성 (vllm 환경명으로 생성)
./create_conda_virtual_env.sh vllm
```

### 1.2. 가상환경 활성화
```bash
# vllm 환경 활성화
conda activate vllm

# 환경 확인
conda info --envs
```

### 1.3. 필수 패키지 설치 확인
```bash
# vLLM 및 PyTorch 버전 확인
python -c "
import vllm
import torch
print(f'vLLM Version: {vllm.__version__}')
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
"
```

## 2. vLLM Docker 실행

### 2.1. HuggingFace 토큰 설정
```bash
# HuggingFace 토큰 환경변수 설정
export HF_TOKEN="your_huggingface_token_here"
```

### 2.2. Docker 컨테이너 실행
```bash
# Mixtral-8x7B 모델로 vLLM 서버 실행
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 8
```

### 2.3. Docker 컨테이너 상태 확인
```bash
# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 로그 확인
docker logs <container_id>

# vLLM 버전 확인
docker exec -it <container_id> python3 -c "import vllm; print(f'vLLM Version: {vllm.__version__}')"
```

### 2.4. API 서버 상태 확인
```bash
# 서버 상태 확인
curl http://localhost:8000/health

# 사용 가능한 모델 확인
curl http://localhost:8000/v1/models

# vLLM 버전 확인
curl http://localhost:8000/version
```

## 3. Python 벤치마크 테스트 실행

### 3.1. 테스트 파일 실행 (화면 출력 + 파일 저장)
```bash
# 디렉토리 이동
cd ~/lab/13-vllm-mistral-8-7B-p4d

# vllm 환경 활성화
conda activate vllm

# 실시간 출력과 파일 저장
python -u 01_mistral__online_test.py 2>&1 | tee mistral_benchmark_$(date +%Y%m%d_%H%M%S).log
```

### 3.2. 백그라운드 실행 (장시간 테스트)
```bash
# 백그라운드 실행
nohup python -u 01_mistral__online_test.py > mistral_benchmark_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 실시간 로그 확인
tail -f mistral_benchmark_*.log
```

### 3.3. Screen 세션 사용 (권장)
```bash
# screen 세션 생성
screen -S mistral_test

# vllm 환경 활성화
conda activate vllm

# screen 내에서 실행
python -u 01_mistral__online_test.py 2>&1 | tee mistral_benchmark_$(date +%Y%m%d_%H%M%S).log

# Ctrl+A, D로 세션 분리
# screen -r mistral_test로 다시 접속
```

## 4. 문제 해결

### 4.1. Conda 환경 문제
```bash
# 가상환경 재생성
conda deactivate
conda env remove -n vllm
./create_conda_virtual_env.sh vllm
conda activate vllm
```

### 4.2. Docker 컨테이너 문제
```bash
# 컨테이너 중지
docker stop <container_id>

# 컨테이너 제거
docker rm <container_id>

# 이미지 재다운로드
docker pull vllm/vllm-openai:latest
```

### 4.3. 포트 충돌 문제
```bash
# 포트 사용 확인
netstat -tulpn | grep 8000

# 다른 포트로 실행
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
    -p 8001:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 8
```

### 4.4. GPU 메모리 부족
```bash
# GPU 사용량 확인
nvidia-smi

# 텐서 병렬 크기 줄이기
--tensor-parallel-size 4  # 8에서 4로 변경
```

## 5. 성능 모니터링

### 5.1. GPU 모니터링
```bash
# 실시간 GPU 사용량 확인
watch -n 1 nvidia-smi

# GPU 사용량 로그
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### 5.2. 시스템 리소스 모니터링
```bash
# CPU 및 메모리 사용량
htop

# 디스크 사용량
df -h
```
```

이제 conda 가상환경 생성과 활성화 과정이 포함되었습니다!