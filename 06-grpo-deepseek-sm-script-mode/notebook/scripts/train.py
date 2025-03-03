
################################################
# 1. Check Cuda version
################################################

import os
import subprocess
import sys

def check_cuda_version():
    print("===== CUDA 환경 진단 정보 =====")
    
    # 시스템 CUDA 버전 확인
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print("NVCC 버전 정보:")
        print(nvcc_output)
    except Exception as e:
        print(f"NVCC 버전 확인 실패: {e}")
    
    # nvidia-smi로 드라이버 및 CUDA 버전 확인
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print("NVIDIA-SMI 정보:")
        print(nvidia_smi_output)
    except Exception as e:
        print(f"NVIDIA-SMI 실행 실패: {e}")
    
    # 환경 변수 확인
    cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    print(f"CUDA_HOME/CUDA_PATH 환경 변수: {cuda_path}")
    
    # LD_LIBRARY_PATH 확인 (리눅스/맥)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    print(f"LD_LIBRARY_PATH: {ld_library_path}")
    
    # Python 버전 확인
    print(f"Python 버전: {sys.version}")
    
    # 설치된 패키지 버전 확인
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
        print("설치된 관련 패키지:")
        for line in pip_freeze.split("\n"):
            if any(pkg in line.lower() for pkg in ["torch", "cuda", "nvidia", "transformers", "vllm", "trl"]):
                print(line)
    except Exception as e:
        print(f"패키지 버전 확인 실패: {e}")
    
    print("==============================")

# 진단 코드 실행
check_cuda_version()

##############################################################################
#  2. Check out torch and cuda version
##############################################################################
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

##############################################################################
# 3. Check out python packages version 
##############################################################################
try:
    result = subprocess.run("pip list | grep -E 'torch|transformers|vllm|trl|datasets'", 
                            shell=True, 
                            capture_output=True, 
                            text=True)
    print("Installed versions:")
    print(result.stdout)
except Exception as e:
    print(f"Error checking versions: {e}")


################################################
# 4. Load basic library and define basic functions
################################################

# Python Built-Ins:
import argparse
import os
import sys

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


################################################
# 5. Define main function
################################################

def main(args):
    # Load and prep dataset

    dataset = get_gsm8k_questions()
    print("## dataset: ", dataset)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    num_generations = args.num_generations

    output_dir="outputs/Qwen-0.5B-GRPO"
    run_name="Qwen-0.5B-GRPO-gsm8k"

    print("## GRPC Config: ")
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        # per_device_train_batch_size=2, # error due to 117 data size
        gradient_accumulation_steps=4,
        num_generations = num_generations,
        # num_generations=16, # success on ml.p4de.24xlarge
        # num_generations=4,  # success on ml.p4d.24xlarge
        # num_generations=6,  # failure on ml.p4d.24xlarge
        # num_generations=8,  # failuer on ml.p4d.24xlarge
        # num_generations=2,  # success on ml.g5.12xlarge
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=.3,
        # vllm_gpu_memory_utilization=.6,
        vllm_device="cuda:0",
        report_to="none" #I'm disabling Wandb.
    )

    print("## Loading model: ")
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=None
            ).to("cuda")

    print("## Loading tokenizer: ")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("## Loading trainer: ")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        #peft_config=peft_config
    )
    print("## Start training: ")
    trainer.train()    

    # Save the model output
    trainer.save_model(args.model_dir)



def parse_args():
    """Parse hyperparameters and data args from CLI arguments and environment variables"""
    parser = argparse.ArgumentParser()

    ##############################################################################
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    ##############################################################################
    # Placeholder
    parser.add_argument("--num_generations", type=int, default=2)

    ##############################################################################
    # Data, model, and output folders are set by combination of CLI args and env vars:
    ##############################################################################
    # parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    # parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    
    args, _ = parser.parse_known_args()
    return args


import subprocess

if __name__ == "__main__":

    # Load job parameters:
    args = parse_args()
    print("## args: \n", args)
    main(args)



    