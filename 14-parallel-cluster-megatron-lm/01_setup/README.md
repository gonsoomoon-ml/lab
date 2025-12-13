# Parallel Cluster Megatron-LM 실습 환경

이 프로젝트는 AWS ParallelCluster를 사용하여 Megatron-LM을 실행하기 위한 실습 환경을 제공합니다.

## 📋 목차

- [개요](#개요)
- [사전 요구사항](#사전-요구사항)
- [설치 가이드](#설치-가이드)
- [사용법](#사용법)
- [문제 해결](#문제-해결)

## 🎯 개요

AWS ParallelCluster는 AWS에서 고성능 컴퓨팅(HPC) 클러스터를 쉽게 배포하고 관리할 수 있게 해주는 오픈소스 클러스터 관리 도구입니다. 이 실습에서는 ParallelCluster를 사용하여 Megatron-LM과 같은 대규모 언어 모델을 효율적으로 훈련하고 추론할 수 있는 환경을 구성합니다.

### 주요 기능
- AWS ParallelCluster 3.11.1 기반 클러스터 구성
- Megatron-LM 모델 훈련 및 추론 환경
- 고성능 컴퓨팅 리소스 활용
- 확장 가능한 분산 학습 환경

## �� 사전 요구사항

### AWS 계정 및 권한
- AWS 계정이 필요합니다
- 다음 AWS 서비스에 대한 접근 권한이 필요합니다:
  - Amazon EC2
  - Amazon VPC
  - Amazon S3
  - AWS CloudFormation
  - AWS IAM

### 시스템 요구사항
- Python 3.10.14
- Conda 또는 Miniconda
- 인터넷 연결

## �� 설치 가이드

### 1. 저장소 클론

```bash
git clone https://github.com/gonsoomoon-ml/Self-Study-Generative-AI.git
cd Self-Study-Generative-AI/lab/14-parallel-cluster-megatron-lm
```

### 2. 가상환경 생성

```bash
cd 01_setup
./create_conda_virtual_env.sh megatron-lm
```

이 스크립트는 다음 작업을 수행합니다:
- Python 3.10.14 기반 가상환경 생성
- AWS ParallelCluster 설치
- Jupyter Kernel 등록

### 3. 설치 확인

```bash
# 가상환경 활성화
source activate megatron-lm

# 설치된 패키지 확인
pip list | grep parallelcluster

# Jupyter Kernel 확인
jupyter kernelspec list
```

## �� 사용법

### 가상환경 관리

```bash
# 가상환경 활성화
source activate megatron-lm

# 가상환경 비활성화
conda deactivate

# 가상환경 목록 확인
conda env list

# 가상환경 삭제
conda env remove -n megatron-lm
```

### Jupyter Kernel 관리

```bash
# Jupyter Kernel 목록 확인
jupyter kernelspec list

# 특정 Kernel 삭제
jupyter kernelspec uninstall -y megatron-lm
```

## �� 문제 해결

### 일반적인 문제들

#### 1. 가상환경 생성 실패
- Conda가 올바르게 설치되어 있는지 확인
- 충분한 디스크 공간이 있는지 확인

#### 2. 패키지 설치 실패
- 인터넷 연결 상태 확인
- pip 업그레이드 후 재시도

#### 3. Jupyter Kernel 등록 실패
- ipykernel이 올바르게 설치되었는지 확인
- 사용자 권한 확인

### 로그 확인

```bash
# 설치 과정에서 발생한 오류 확인
conda info --envs
which python
pip list
```

## 📚 추가 리소스

- [AWS ParallelCluster 공식 문서](https://docs.aws.amazon.com/parallelcluster/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [AWS HPC 워크샵](https://www.hpcworkshops.com/)

## �� 기여하기

이 프로젝트에 기여하고 싶으시다면:
1. 이슈를 생성하여 버그나 개선사항을 보고
2. Pull Request를 통해 코드 개선을 제안

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👨‍�� 작성자

Gonsoo Moon - [GitHub](https://github.com/gonsoomoon-ml)

---

**마지막 업데이트**: 2024년 10월
