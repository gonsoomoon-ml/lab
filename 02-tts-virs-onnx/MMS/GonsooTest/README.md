

# MMS-STT-Test
- https://k2-fsa.github.io/sherpa/onnx/tts/mms.html?highlight=convert+tts+onnx


# A. Setup Environment

## 1. Use EC2 ( CPU or GPU, ml.g4dn.xlarge)


## 2. Downlaod Git
Clone the folloiwng git
```
git clone https://github.com/gonsoomoon-ml/lab.git
```
## 3. Create conda virtual environment
- install conda
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
    rm ~/miniconda.sh
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
    conda --version
    ```
- create conda virtual environment
    ```
    conda create -n onnx-conda-py38 python=3.8
    source activate onnx-conda-py38
    ```
- install other dependency
    ```
    python -m pip install -U pip
    pip install -qq onnx scipy Cython
    pip install -qq torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```
- install Sherpa-onnx, [guide](https://k2-fsa.github.io/sherpa/onnx/install/linux.html)
    ```
    git clone https://github.com/k2-fsa/sherpa-onnxcd sherpa-onnx
    mkdir buildcd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j6
    ```
- install jupyter kernel
    ```
    # Install Jupyter notebook kernel
    pip install ipykernel 
    python -m ipykernel install --user --name onnx-conda-py38  --display-name "onnx-conda-py38"

    # For reference:
    # jupyter kernelspec list
    # jupyter kernelspec uninstall VITS
    ``` 

# B. Test Compilation
## 0. Obtain model weight file
-  run the commands as below, [guide link](https://k2-fsa.github.io/sherpa/onnx/tts/mms.html?highlight=convert+tts+onnx)
    ``` 
    name=eng
    wget -q https://huggingface.co/facebook/mms-tts/resolve/main/models/$name/G_100000.pth
    wget -q https://huggingface.co/facebook/mms-tts/resolve/main/models/$name/config.json
    wget -q https://huggingface.co/facebook/mms-tts/resolve/main/models/$name/vocab.txt 
    ```    
## 1. Run test notebook
- run /home/ubuntu/lab/02-stt-virs-onnx/MMS/GonsooTest/mms-onnx.ipynb
- if you successfully run it, model.onnx is generated. 

## 2. Using model.onnx, generative wave file based on text 
- Run the following command
    ```    
    sherpa-onnx-offline-tts \
    --vits-model=./model.onnx \
    --vits-tokens=./tokens.txt \
    --debug=1 \
    --output-filename=./mms-eng.wav \
    "How are you doing today? This is a text-to-speech application using models from facebook with next generation Kaldi"
    
    sherpa-onnx-offline-tts \
    --vits-model=./model.onnx \
    --vits-tokens=./tokens.txt \
    --debug=1 \
    --output-filename=./mms-eng.wav \
    "How are you doing today?"  
    
    sherpa-onnx-offline-tts \
    --vits-model=./model.onnx \
    --vits-tokens=./tokens.txt \
    --debug=1 \
    --output-filename=./mms-eng.wav \
    "Hi Gonsoo?"    
    
    sherpa-onnx-offline-tts \
    --vits-model=./model.onnx \
    --vits-tokens=./tokens.txt \
    --debug=1 \
    --output-filename=./mms-eng.wav \
    "How are you doing today? This is a text-to-speech application using models from facebook with next generation Kaldi, Itis a long sequence test by Gonsoo"  
    
    ```    