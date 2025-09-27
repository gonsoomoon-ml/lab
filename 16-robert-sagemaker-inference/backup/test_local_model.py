#!/usr/bin/env python3
"""
로컬 환경에서 klue/roberta-base 모델 테스트
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

def test_klue_roberta():
    """KLUE RoBERTa 모델 로드 및 테스트"""

    print("=" * 60)
    print("KLUE RoBERTa 모델 테스트")
    print("=" * 60)

    # 1. 모델과 토크나이저 로드
    print("\n1. 모델 로딩 중...")
    model_name = "klue/roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # GPU 사용 가능 시 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   사용 디바이스: {device}")
    model = model.to(device)

    print(f"   ✓ 모델 로드 완료: {model_name}")

    # 2. 테스트 문장 준비
    test_sentences = [
        "김치찌개와 된장찌개는 한국의 대표 음식입니다.",
        "인공지능 기술이 빠르게 발전하고 있습니다.",
        "서울은 대한민국의 수도입니다."
    ]

    print("\n2. 테스트 문장 인코딩...")
    for i, sentence in enumerate(test_sentences):
        print(f"   [{i+1}] {sentence}")

    # 3. 토크나이징
    print("\n3. 토크나이징...")
    inputs = tokenizer(
        test_sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # GPU로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"   입력 shape: {inputs['input_ids'].shape}")
    print(f"   ✓ 토크나이징 완료")

    # 4. 추론 수행
    print("\n4. 모델 추론 중...")
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    inference_time = (time.time() - start_time) * 1000
    print(f"   추론 시간: {inference_time:.2f}ms")

    # 5. Mean Pooling (문장 임베딩 생성)
    print("\n5. Mean Pooling 적용...")

    # Attention mask를 사용한 mean pooling
    hidden_states = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1)

    # 마스킹된 부분을 제외하고 평균 계산
    masked_hidden_states = hidden_states * attention_mask
    sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
    sum_attention_mask = torch.sum(attention_mask, dim=1)

    sentence_embeddings = sum_hidden_states / sum_attention_mask

    print(f"   임베딩 shape: {sentence_embeddings.shape}")
    print(f"   ✓ 문장 임베딩 생성 완료")

    # 6. 코사인 유사도 계산
    print("\n6. 문장 간 유사도 계산...")

    # 정규화
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    # 코사인 유사도 계산
    similarity_matrix = torch.matmul(sentence_embeddings, sentence_embeddings.T)

    print("\n   유사도 매트릭스:")
    for i in range(len(test_sentences)):
        for j in range(len(test_sentences)):
            print(f"   [{i},{j}]: {similarity_matrix[i,j].item():.4f}", end="  ")
        print()

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

    return model, tokenizer

def test_inference_speed(model, tokenizer, num_iterations=10):
    """추론 속도 테스트"""

    print("\n" + "=" * 60)
    print("추론 속도 벤치마크")
    print("=" * 60)

    device = next(model.parameters()).device
    test_text = "이것은 추론 속도 테스트를 위한 문장입니다."

    # Warm up
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model(**inputs)

    # 실제 테스트
    times = []
    for i in range(num_iterations):
        start_time = time.time()

        with torch.no_grad():
            _ = model(**inputs)

        elapsed_time = (time.time() - start_time) * 1000
        times.append(elapsed_time)
        print(f"   Iteration {i+1:2d}: {elapsed_time:.2f}ms")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n   평균 추론 시간: {avg_time:.2f}ms (±{std_time:.2f}ms)")
    print(f"   최소 시간: {min(times):.2f}ms")
    print(f"   최대 시간: {max(times):.2f}ms")

    return avg_time

if __name__ == "__main__":
    # 모델 테스트
    model, tokenizer = test_klue_roberta()

    # 속도 벤치마크
    test_inference_speed(model, tokenizer, num_iterations=10)