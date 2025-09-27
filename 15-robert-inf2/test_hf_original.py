#!/usr/bin/env python3

import torch
from transformers import AutoModel, AutoTokenizer
import time

# HuggingFace 원본 모델
MODEL_NAME = "klue/roberta-base"

def load_hf_model():
    """HuggingFace 원본 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

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

def cosine_similarity(query_emb, doc_embs):
    """코사인 유사도 계산"""
    query_norm = query_emb / query_emb.norm(dim=-1, keepdim=True)
    doc_norms = doc_embs / doc_embs.norm(dim=-1, keepdim=True)
    return torch.matmul(query_norm, doc_norms.T).squeeze(0)

def main():
    print("=" * 60)
    print("HuggingFace 원본 모델 테스트")
    print("=" * 60)

    tokenizer, model = load_hf_model()

    # 동일한 테스트 데이터
    query = "맛있는 한국 음식"

    documents = [
        "김치찌개와 된장찌개는 한국의 대표 음식입니다.",
        "서울 강남의 유명한 한식당을 소개합니다.",
        "일본 라멘과 초밥은 인기가 많습니다.",
        "파리의 에펠탑은 프랑스의 상징입니다.",
        "삼성전자 주가가 상승했습니다.",
        "양자 컴퓨터의 발전이 가속화되고 있습니다.",
    ]

    print(f"\n쿼리: '{query}'")
    print("-" * 60)

    # 인코딩
    start = time.time()
    query_emb = encode_hf([query], tokenizer, model)
    doc_embs = encode_hf(documents, tokenizer, model)
    encode_time = (time.time() - start) * 1000

    # 유사도 계산
    similarities = cosine_similarity(query_emb, doc_embs)

    # 결과 출력
    print(f"\n인코딩 시간: {encode_time:.1f}ms\n")
    print("HuggingFace 모델 유사도 결과:")
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