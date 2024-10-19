import matplotlib.pyplot as plt
import IPython.display as ipd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("./configs/ljs_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("models/pretrained_ljs.pth", net_g, None)


import torch
import torch_neuron

# Convert scalar value to tensor
noise_scale = torch.tensor(0.667)
noise_scale_w = torch.tensor(0.8)
length_scale = torch.tensor(1.0)

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

stn_tst = get_text("VITS is Awesome!", hps)
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])

# use torch_neuron.trace
print("## Starting Compilation")
traced_model = torch_neuron.trace(wrapped_model, 
                                  (x_tst, x_tst_lengths, noise_scale, noise_scale_w, length_scale))


print("## Finishing Compilation")
# # Save traced model 
# torch.jit.save(traced_model, "traced_vits_model_neuron.pt")


# # inference 
# traced_output = traced_model(x_tst, x_tst_lengths, noise_scale, noise_scale_w, length_scale)
# print("output shape:", traced_output[0].shape)