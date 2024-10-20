

# TTS VITS Model on NVidia Triton Docker

# Summary
-  ["Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"](https://github.com/jaywalnut310/vits.git) 모델을 NVidia Triton Docker 에서 모델 서빙하는 코드 입니다.

---

# A. Setup environment

## 1. Launch SageMaker notebook instance 
- 아래에서 테스트 되었습니다.
    - ml.g4dn.xlarge
    - ml.g5.xlarge

## 2. Download git
Clone the folloiwng git
```
git clone https://github.com/gonsoomoon-ml/lab.git
```
## 3. Create conda virtual environment
- create conda virtual environment
    ```
    conda create -n conda-vits-py310 python=3.10
    source activate conda-vits-py310
    ```
- Clone VITS Git repo
    ```
    cd /home/ec2-user/SageMaker/lab/00-trition-tts-vits/02-tts-vits-docker-trition
    git clone https://github.com/jaywalnut310/vits.git
    ```
- install requirements.txt
    ```
    pip install -r setup/requirements.txt
    ```
- install other dependency
    ```
    sudo apt-get install espeak
    cd /home/ec2-user/SageMaker/lab/00-trition-tts-vits/02-tts-vits-docker-trition/vits/monotonic_align
    python setup.py build_ext --inplace
    ```
- install jupyter kernel
    ```
    # Install Jupyter notebook kernel
    pip install ipykernel 
    python -m ipykernel install --user --name conda-vits-py310  --display-name "conda-vits-py310"

    # For reference:
    # jupyter kernelspec list
    # jupyter kernelspec uninstall VITS
    ``` 
- Verify the dependency
    ``` 
    pip list | grep -E "setuptools|Cython|librosa|matplotlib|numba|numpy|phonemizer|scipy|torch|Unidecode"
    ``` 
- It is okay if you have the follwoing information:
    ``` 
    ``` 

# B. Download Pre-Trained Models
- [Important] Downlaod model links as below to /home/ec2-user/SageMaker/lab/00-trition-tts-vits/02-tts-vits-docker-trition/vits/models/
    - https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2

# C. Run notebook
- 0.0-create-tts-vits-model.ipynb
    - VITS Model 을 로딩하고, TorchScript 로 변환하고 저장을 함.
- 1.0-serve-tts-vits-trition-docker.ipynb
    - SageMaker Triton Docker 를 다운로드 맏고, Triton Docker 에서 서빙하는 코드 임


