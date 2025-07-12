# Original Code: https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/examples/generation_demo.py

import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

model_path = "/home/ubuntu/model_hf/Llama-3.1-8B/"
traced_model_path = "/home/ubuntu/traced_model/Llama-3.1-8B/"

torch.manual_seed(0)


def run_llama_generate():
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id,
    }
    generation_config.update(**generation_config_kwargs)

    neuron_config = NeuronConfig(
        tp_degree=2,
        batch_size=1,
        max_context_length=32,
        seq_len=64,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=True,
        flash_decoding_enabled=False,
        fused_qkv=True  # 이 줄 추가
    )
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
        
    # Compile and save model.
    print("########################################################")
    print("Compiling and saving model...")
    print("########################################################")
    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)
    
    # Load from compiled checkpoint.
    print("########################################################")
    print("Loading model from compiled checkpoint...")
    print("########################################################")
    model = NeuronLlamaForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("########################################################")
    print("Generating outputs...")
    print("########################################################")
    prompts = ["I believe the meaning of life is", "The color of the sky is"]
    sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("########################################################")
    print("Generated outputs:")
    print("########################################################")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    run_llama_generate()
