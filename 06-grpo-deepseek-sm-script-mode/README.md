# Lab: QWEN 0.5B 를 베이스 모델로 QRPO 강화 학습 배우기

## 1. 배경 및 목적

이 워크샵은 [QRPO](https://medium.com/data-science-in-your-pocket/what-is-grpo-the-rl-algorithm-used-to-train-deepseek-12acc19798d3) 강화 학습 알고리즘 및 gsm8k 수학 데이터셋 사용 및 Qwen-0.5b 를 베이스 모델로 하여 강화 학습을 합니다. 아래의 참조 노트북을 기반으로 SageMaker 에서 학습을 
재구성한 내용 입니다.

실제로 학습을 자세히 살펴 보기 위해 "훈련 데이터", "QRPO 학습 과정", "리워드 함수 및 제공 내용" 을 확인하면서 QRPO 강화학습을 배울 수 있습니다.
- 참조: [Original Notebook](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing#scrollTo=Q7qTZbUcg5VD)


## 2. 모델 훈련
### 2.1 훈련 데이터 포맷팅
```
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "72",
  "prompt": [
    {
      "content": "\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n",
      "role": "system"
    },
    {
      "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
      "role": "user"
    }
  ]
}
```

### 2.2 QRPO 강화 학습
#### 2.2.1 Reward functions
- 다음은 QRPO 리워드 함수 중에서 correctness 에 대한 보상 함수의 예 입니다.
    - 보상 함수의 예시 처럼, 주어진 질문에 정확한 답을 제공하면 2.0 을 리워드로 제공하고, 틀리면 0.0 을 제공 합니다.

    ```
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    ```

#### 2.2.2 QRPO 학습 진행
아래는 일부 훈련 데이터가 제공되고, vllm 이 "질문" 을 받아서 QWEN 0.5B 의 베이스 모델로 부터 "응답"을 얻은 후에, 실제 "정답" 을 얻은 것을 포맷팅 해서 제공 합니다. 

    ```
    -------------------- Question:
    Tom reads 10 hours over 5 days.  He can read 50 pages per hour.  Assuming he reads the same amount every day how many pages does he read in 7 days? 
    Answer:
    700 
    Response:
    <reasoning>
    Tom reads 10 hours over 5 days, so he reads 10 * 10 = 100 pages in total. He reads 50 pages per hour, so in 5 days, he reads 50 * 5 = 250 pages. Therefore, he reads 250 * 7 = 1750 pages in 7 days.
    </reasoning>
    <answer>
    1750
    </answer> 
    Extracted:
    1750

    -------------------- Question:
    A new movie gets released and makes $120 million in the box office for its opening weekend.  It ends up making 3.5 times that much during its entire run.  If the production company gets to keep 60%, how much profit did they make if the movie cost $60 million to produce? 
    Answer:
    192000000 
    Response:
    <reasoning>
    The movie makes $120 million in the first week and $360 million in total during its run, resulting in a profit of $240 million. If the production company gets to keep 60%, they profit of 60% of the total revenue.
    </reasoning>
    <answer>
    The profit the production company made is 60% of $240 million, which is $144 million.
    </answer> 
    Extracted:
    The profit the production company made is 60% of $240 million, which is $144 million.
    ```
#### 2.2.3 QRPO 의 학습 메트릭 변화
아래는 학습이 진행되면서,중간 중간의 학습 메트릭의 값 및 , 보상값에 대한 내용을 보여 줍니다.

    ```
    {'loss': 0.006, 'grad_norm': 6.5625, 'learning_rate': 4.692566002491917e-06, 'completion_length': 123.34375, 'rewards/xmlcount_reward_func': 0.17096875421702862, 'rewards/soft_format_reward_func': 0.0, 'rewards/strict_format_reward_func': 0.0, 'rewards/int_reward_func': 0.15625, 'rewards/correctness_reward_func': 0.375, 'reward': 0.7022187635302544, 'reward_std': 0.7123659029603004, 'kl': 0.15093354508280754, 'epoch': 0.24}

    ....

    {'loss': 0.0066, 'grad_norm': 2.03125, 'learning_rate': 0.0, 'completion_length': 120.779296875, 'rewards/xmlcount_reward_func': 0.2301914133131504, 'rewards/soft_format_reward_func': 0.0, 'rewards/strict_format_reward_func': 0.0, 'rewards/int_reward_func': 0.400390625, 'rewards/correctness_reward_func': 0.80078125, 'reward': 1.4313633143901825, 'reward_std': 0.7792688012123108, 'kl': 0.16573260724544525, 'epoch': 1.0}
    ```

## 3. 실습
### 3.1 실험 환경
- 노트북 실험 환경
    - SageMaker Studio 의 [Code Editor](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/code-editor.html) 의 ml.g5.12xlarge 를 사용하였습니다.
- SageMaker Training Instance 중에 하나를 준비
    - ml.g5.12xlarge 
    - ml.p4d.24xlarge 
    - ml.p4de.24xlarge 

### 3.2 실행 순서
- Step1: Git Clone
    ```
    git clone https://github.com/gonsoomoon-ml/lab.git
    ```
- Step2: 가상 환경 설치
    ```
    cd /home/sagemaker-user/lab/06-grpo-deepseek-sm-script-mode/setup
    ./create_conda_virtual_env.sh  qrpo-sm-training
    ```
- Step3: 노트북 실행 
    - notebook/01-load-dataset.ipynb
        - 훈련에 사용될 훈련 데이터셋을 로드하고, Message 형태로 데이터 포맷팅을 하는 방법을 보여 줌.
    - notebook/02-local-qwen-05b-grpo.ipynb
        - 훈련 코드읜 train.py 를 로컬에서 실행
    - notebook/03-sm-qwen-05b-grpo.ipynb
        - 훈련 코드읜 SageMaker Training Job 으로 클라우드에서 실행





