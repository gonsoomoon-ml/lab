import os
os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_COMPILE_CACHE_URL'] = "/workspace/neuron_cache"
os.environ['NEURON_CC_FLAGS'] = "--cache_dir=/workspace/neuron_cache"
os.environ['NEURON_COMPILED_ARTIFACTS_PATH'] = "/workspace/local-models/TinyLlama/TinyLlama-1.1B-Chat-v1.0/neuron-compiled-artifacts/302fdb8c07ace8605c4430094da27814"

print("Starting vLLM Neuron inference...")

try:
    from vllm import LLM, SamplingParams
    print("vLLM imported successfully")
    
    print("Loading model...")
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=8,
        max_model_len=128,
        device="neuron",
        tensor_parallel_size=2,
        download_dir="/workspace/local-models",
        override_neuron_config={
            "compiled_artifacts_path": "/workspace/local-models/TinyLlama/TinyLlama-1.1B-Chat-v1.0/neuron-compiled-artifacts/302fdb8c07ace8605c4430094da27814"
        })
    print("Model loaded successfully")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # note that top_k must be set to lower than the global_top_k defined in
    # the neuronx_distributed_inference.models.config.OnDeviceSamplingConfig
    sampling_params = SamplingParams(top_k=10, temperature=0.8, top_p=0.95)

    print("Generating text...")
    outputs = llm.generate(prompts, sampling_params)
    print("Text generation completed")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()