# The git reference: https://github.com/jaywalnut310/vits.git
# Run the command in console in conda virtual environment
# time python test_compile_neuron.py 2>&1 | tee compile.out

import sys
import os

# For using custom library
sys.path.append(os.path.abspath("../"))

import os
import json
import math
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

# from scipy.io.wavfile import write


# get hparams
hps = utils.get_hparams_from_file("../configs/ljs_base.json")

# Load STT VITS model
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

# Load weight file
_ = utils.load_checkpoint("../models/pretrained_ljs.pth", net_g, None)

import torch
import torch_neuron
# from torch_neuron import analyze_model

# wrapping model
class VITSWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, noise_scale, noise_scale_w, length_scale):
        return self.model.infer(x, x_lengths, 
                                noise_scale=noise_scale.item(),
                                noise_scale_w=noise_scale_w.item(), 
                                length_scale=length_scale.item())

wrapped_model = VITSWrapper(net_g)

# create dummy input
stn_tst = torch.randint(low=1, high=10, size=(33,), dtype=torch.int64)
x_tst = stn_tst.unsqueeze(0)
x_tst_lengths = torch.LongTensor([stn_tst.size(0)])

# convert scalar value to tensor
noise_scale = torch.tensor(0.667)
noise_scale_w = torch.tensor(0.8)
length_scale = torch.tensor(1.0)

print("x_tst: ", x_tst)
print("x_tst shape: ", x_tst.shape)
print("x_tst_length shape: ", x_tst_lengths.shape)
print("x_tst_length: ", x_tst_lengths)
    
# use torch_neuron.trace
print("--" * 50)
print("## Starting Compilation")
print("--" * 50)
traced_model = torch_neuron.trace(wrapped_model, 
                                  (x_tst, x_tst_lengths, 
                                   noise_scale, noise_scale_w, length_scale))


print("## Finishing Compilation")


