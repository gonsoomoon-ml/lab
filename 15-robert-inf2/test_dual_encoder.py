#!/usr/bin/env python3

import torch
import torch_neuronx
from transformers import AutoTokenizer
import time
import os

# 설정
QUERY_ENCODER_PATH = "/home/ubuntu/data/query-encoder-neuron"
ORIGINAL_MODEL = "klue/roberta-base"

def load_model():
    """모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    model_path = os.path.join(QUERY_ENCODER_PATH, "model.neuron")
    encoder = torch.jit.load(model_path)
    encoder.eval()
    return tokenizer, encoder

def encode(texts, tokenizer, encoder):
    """텍스트를 임베딩으로 변환 (Mean Pooling)"""
    # 배치 크기 8에 맞게 패딩
    padded = texts + [""] * (8 - len(texts))

    inputs = tokenizer(padded, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=128)

    with torch.no_grad():
        outputs = encoder(inputs['input_ids'], inputs['attention_mask'])
        # last_hidden_state로 mean pooling (pooler 대신)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (hidden_states * mask).sum(1) / mask.sum(1)

    return embeddings[:len(texts)]

def cosine_similarity(query_emb, doc_embs):
    """코사인 유사도 계산"""
    query_norm = query_emb / query_emb.norm(dim=-1, keepdim=True)
    doc_norms = doc_embs / doc_embs.norm(dim=-1, keepdim=True)
    return torch.matmul(query_norm, doc_norms.T).squeeze(0)

def main():
    tokenizer, encoder = load_model()

    # 쿼리
    query = "맛있는 한국 음식"

    # 문서들 (유사도 높음 → 낮음)
    documents = [
        "김치찌개와 된장찌개는 한국의 대표 음식입니다.",        # 매우 관련
        "서울 강남의 유명한 한식당을 소개합니다.",             # 관련
        "일본 라멘과 초밥은 인기가 많습니다.",                # 약간 관련 (음식)
        "파리의 에펠탑은 프랑스의 상징입니다.",               # 무관 (관광)
        "삼성전자 주가가 상승했습니다.",                     # 완전 무관 (경제)
        "양자 컴퓨터의 발전이 가속화되고 있습니다.",          # 완전 무관 (기술)
    ]

    print("=" * 60)
    print(f"쿼리: '{query}'")
    print("=" * 60)

    # 인코딩
    start = time.time()
    query_emb = encode([query], tokenizer, encoder)
    doc_embs = encode(documents, tokenizer, encoder)
    encode_time = (time.time() - start) * 1000

    # 유사도 계산
    similarities = cosine_similarity(query_emb, doc_embs)

    # 결과 출력
    print(f"\n인코딩 시간: {encode_time:.1f}ms\n")
    print("유사도 결과:")
    print("-" * 60)

    for idx, (doc, score) in enumerate(zip(documents, similarities), 1):
        bar = "█" * int(score * 20)
        print(f"{idx}. [{score:.3f}] {bar}")
        print(f"   {doc}\n")

    # 상위 3개
    print("=" * 60)
    print("상위 3개 문서:")
    top_k = torch.topk(similarities, 3)
    for rank, (idx, score) in enumerate(zip(top_k.indices, top_k.values), 1):
        print(f"  {rank}. [{score:.3f}] {documents[idx]}")

if __name__ == "__main__":
    main()