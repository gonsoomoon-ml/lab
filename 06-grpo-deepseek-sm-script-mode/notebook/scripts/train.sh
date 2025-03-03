#!/bin/bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.48.3 vllm==0.7.2 trl==0.14.0 datasets==3.2.0
python train.py num_generations 4