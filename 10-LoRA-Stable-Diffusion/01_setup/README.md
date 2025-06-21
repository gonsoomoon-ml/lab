# 설치 가이드

## 1. 실습환경
- 이 노트북은 [SageMaker AI Studio Jupyterlab](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/studio-updated-jl.html) 에서 테스트 완료 되었습니다.
    - IAM role : SageMaker AI Studio 를 생성 및 생성 프로파일에는 아래와 같은 권한이 추가 되어 있어야 합니다.
        - ![IAM_role_permission.png](img/IAM_role_permission.png)
    - 환경: ml.m5.xlarge 
    - Region : us-west-2
- Amazon Bedrock Model Access 가 필요 합니다.
    - Amazon Nova Pro
    - Amazon Nova lite
    - Amazon Nova micro
    - Claude 3.7 Sonnet 
    - Claude 3.5 Sonnet 
    - Claude 3.5 Haiku
    - Titan Embeddings G1 – Text
    

## 2. 실습 환경 세팅
### 2.1. JupyerLab 을 열고 아래와 같이 터미널을 오픈 하세요.
- ![open_terminal.png](img/open_terminal.png)

### 2.2. 아래와 같이 명령어를 넣고 Git 리포를 클로닝 합니다.
#### [중요] 현재는 아래 git 을 다운로드 해서 사용을 해야 합니다.
```bash
pwd
git clone https://github.com/gonsoomoon-ml/Self-Study-Generative-AI.git
```
- ![git_clone.png](img/git_clone.png)

### 2.3. Conda 설치 파일 다운로드
**중요**: 대용량 설치 파일들은 GitHub 제한으로 인해 별도로 다운로드해야 합니다.

#### Anaconda3 설치 파일 다운로드:
```bash
cd /home/sagemaker-user/Self-Study-Generative-AI/lab/10-LoRA-Stable-Diffusion/01_setup
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
```

#### Miniconda3 설치 파일 다운로드:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
```

### 2.4. Conda Virtual Environment 생성 (약 1분 소요)
- 가상 환경을 구성 합니다.
- 터미널에서 아래 명령어를 실행하여 setup 폴더로 이동을 합니다. 
    ```
    cd /home/sagemaker-user/Self-Study-Generative-AI/lab/10-LoRA-Stable-Diffusion/01_setup
    ```
- shell 을 아래의 명령어를 넣어서 실행 합니다. 가상 환경 이름은 원하는 이름으로 하셔도 좋습니다. 여기서는 pytorch 으로 했습니다.
    ```
    ./create_conda_virtual_env.sh pytorch
    ```    

- 설치 확인을 합니다. 에러가 발생했는지 확인 합니다.

## 설치가 완료 되었습니다. 축하 드립니다. !
- [README 로 다시 이동하기](../README.md)






