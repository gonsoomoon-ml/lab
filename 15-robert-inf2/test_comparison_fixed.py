#!/usr/bin/env python3

import torch
from transformers import AutoModel, AutoTokenizer
import torch_neuronx
import os
import time

# 모델 경로
HF_MODEL = "klue/roberta-base"
NEURON_PATH = "/home/ubuntu/data/query-encoder-neuron"

def load_models():
    """HF와 Neuron 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    # HF 모델
    hf_model = AutoModel.from_pretrained(HF_MODEL)
    hf_model.eval()

    # Neuron 모델
    neuron_model = torch.jit.load(os.path.join(NEURON_PATH, "model.neuron"))
    neuron_model.eval()

    return tokenizer, hf_model, neuron_model

def encode_hf(texts, tokenizer, model):
    """HF 모델로 인코딩 (Mean Pooling)"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling (pooler 대신)
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (embeddings * mask).sum(1) / mask.sum(1)

    return embeddings

def encode_neuron(texts, tokenizer, model):
    """Neuron 모델로 인코딩 (Mean Pooling)"""
    # 배치 크기 8에 맞춤
    padded = texts + [""] * (8 - len(texts))
    inputs = tokenizer(padded, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        # last_hidden_state로 mean pooling
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (hidden_states * mask).sum(1) / mask.sum(1)

    return embeddings[:len(texts)]

def cosine_similarity(query_emb, doc_embs):
    """코사인 유사도"""
    query_norm = query_emb / query_emb.norm(dim=-1, keepdim=True)
    doc_norms = doc_embs / doc_embs.norm(dim=-1, keepdim=True)
    return torch.matmul(query_norm, doc_norms.T).squeeze(0)

def test_model(name, encode_fn, tokenizer, model):
    """모델 테스트"""
    query = "맛있는 한국 음식"
    documents = [
        "김치찌개와 된장찌개는 한국의 대표 음식입니다.",
        "서울 강남의 유명한 한식당을 소개합니다.",
        "일본 라멘과 초밥은 인기가 많습니다.",
        "파리의 에펠탑은 프랑스의 상징입니다.",
        "삼성전자 주가가 상승했습니다.",
        "양자 컴퓨터의 발전이 가속화되고 있습니다.",
    ]

    print(f"\n{name} 모델 테스트")
    print("-" * 60)

    start = time.time()
    query_emb = encode_fn([query], tokenizer, model)
    doc_embs = encode_fn(documents, tokenizer, model)
    encode_time = (time.time() - start) * 1000

    similarities = cosine_similarity(query_emb, doc_embs)

    print(f"인코딩 시간: {encode_time:.1f}ms")
    print(f"유사도 범위: {similarities.min():.3f} ~ {similarities.max():.3f}")

    # 순위별 출력
    values, indices = torch.sort(similarities, descending=True)
    for rank, (idx, score) in enumerate(zip(indices, values), 1):
        print(f"  {rank}. [{score:.3f}] {documents[idx][:40]}...")

    return similarities

def main():
    print("=" * 60)
    print("Mean Pooling 방식 비교 (pooler 사용 안함)")
    print("=" * 60)

    tokenizer, hf_model, neuron_model = load_models()

    # 각 모델 테스트
    hf_sim = test_model("HuggingFace", encode_hf, tokenizer, hf_model)
    neuron_sim = test_model("Neuron", encode_neuron, tokenizer, neuron_model)

    # 차이 비교
    print("\n" + "=" * 60)
    print("유사도 차이 비교")
    print("-" * 60)
    diff = (hf_sim - neuron_sim).abs()
    print(f"평균 차이: {diff.mean():.4f}")
    print(f"최대 차이: {diff.max():.4f}")

if __name__ == "__main__":
    main()