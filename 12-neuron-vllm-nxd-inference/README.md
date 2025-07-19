# Amazon Neuron 를 이용한 vLLM 서빙

Last Updated: July 18, 2025


# 1. 사전 환경 셋없
## 1.1.환경
    - TrN1
## Nueron 도커 컨테이너에서 작업하기
* AWS Neuron Deep Learning Containers
    * https://github.com/aws-neuron/deep-learning-containers?tab=readme-ov-file#vllm-inference-neuronx
    * 최근 Docker Version:  vLLM 0.7.2
        * https://github.com/aws-neuron/deep-learning-containers/blob/2.24.1/docker/vllm/inference/0.7.2/Dockerfile.neuronx
        * Docker
            * public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.7.2-neuronx-py310-sdk2.24.1-ubuntu22.04



* Docker 안에서 실행
    * docker run -d --name neuron-vllm \
            -v $(pwd):/workspace \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            -v ~/.cache/neuron:/root/.cache/neuron \
            --privileged public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.7.2-neuronx-py310-sdk2.24.1-ubuntu22.04 \
            tail -f /dev/null
    * docker exec -it neuron-vllm /bin/bash
    * python /workspace/quick_start.py
        # docker exec neuron-vllm python /workspace/quick_start.py


# 2. Example 코드
## 1. 모델 컴파일 및 오프라인 추론
- offline examples
## 2. 모델 컴파일 및 온라인 추론
- online examples