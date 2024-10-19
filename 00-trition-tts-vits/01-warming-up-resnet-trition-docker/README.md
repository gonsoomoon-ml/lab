
# A. Setup Environment
### 1. Create Conda Virtual Environment
```
conda create -n resnet-conda-py310 python=3.10

# Activate Python venv 
# source activate vits-conda
source activate resnet-conda-py310
python -m pip install -U pip

pip install torch==1.13.1 torchvision==0.14.1
pip install -U pip awscli boto3 sagemaker
pip install nvidia-pyindex 
pip install tritonclient[all]
pip install jupyter notebook 


# Install Jupyter notebook kernel
pip install ipykernel 
python -m ipykernel install --user --name resnet-conda-py310  --display-name "resnet-conda-py310"
```

### 2. if there is docker installation on Amazon linux O/S, follow this:
```
sudo amazon-linux-extras install docker
sudo systemctl start docker
sudo systemctl enable docker
```

```
```