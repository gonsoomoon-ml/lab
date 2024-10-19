import torch
import torch_neuron

def load_neuron_model(model_path):
    device = torch.device("cpu")  # 먼저 CPU에 로드
    model = torch.jit.load(model_path, map_location=device)
    return torch_neuron.DataParallel(model)

# 사용 예시
model_path = "traced_vits_model_neuron.pt"
loaded_neuron_model = load_neuron_model(model_path)

# Prepare input
stn_tst = get_text("VITS is Awesome!", hps)

x_tst = stn_tst.unsqueeze(0)
x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
# Convert scalar value to tensor
noise_scale = torch.tensor(0.667)
noise_scale_w = torch.tensor(0.8)
length_scale = torch.tensor(1.0)

# inference
traced_output = loaded_neuron_model(x_tst, x_tst_lengths, noise_scale, noise_scale_w, length_scale)
print("output shape:", traced_output[0].shape)

