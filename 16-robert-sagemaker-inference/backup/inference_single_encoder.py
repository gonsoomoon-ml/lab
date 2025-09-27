#!/usr/bin/env python3
"""
SageMaker Inference Script for KLUE RoBERTa
"""

import os
import json
import logging
import sys
import traceback
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def model_fn(model_dir):
    logger.info(f"Loading model from {model_dir}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        local_model_dir = os.path.join(os.path.dirname(__file__), "model")

        if os.path.exists(os.path.join(local_model_dir, "config.json")):
            model_path = local_model_dir
            logger.info(f"Loading model from local directory: {local_model_dir}")
        elif os.path.exists(os.path.join(model_dir, "config.json")):
            model_path = model_dir
            logger.info(f"Loading model from model_dir: {model_dir}")
        else:
            model_path = "klue/roberta-base"
            logger.info(f"Downloading model from HuggingFace: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

        model = model.to(device)
        model.eval()

        model.tokenizer = tokenizer
        model._device = device

        logger.info("Model loaded successfully")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def input_fn(input_data, content_type):
    logger.info(f"Received content_type: {content_type}")

    try:
        if content_type == 'application/json':
            input_str = input_data
            if isinstance(input_data, bytes):
                input_str = input_data.decode('utf-8')

            data = json.loads(input_str)

            # 입력 형식 검증
            if 'texts' not in data:
                raise ValueError("Input JSON must contain 'texts' field")

            texts = data['texts']
            if not isinstance(texts, list):
                texts = [texts]

            max_length = data.get('max_length', 128)

            return {
                'texts': texts,
                'max_length': max_length
            }
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def predict_fn(data, model):
    logger.info("Starting prediction")

    try:
        tokenizer = model.tokenizer
        device = model._device

        texts = data['texts']
        max_length = data['max_length']

        logger.info(f"Processing {len(texts)} text(s)")

        # 토크나이징
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # GPU로 이동
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 추론
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Mean Pooling
        hidden_states = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1)

        masked_hidden_states = hidden_states * attention_mask_expanded
        sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
        sum_attention_mask = torch.sum(attention_mask_expanded, dim=1)

        sentence_embeddings = sum_hidden_states / sum_attention_mask

        # 정규화
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        result = {
            'embeddings': sentence_embeddings.cpu().numpy().tolist(),
            'embedding_dim': sentence_embeddings.shape[1],
            'num_texts': len(texts)
        }

        logger.info("Prediction completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def output_fn(prediction, accept):
    logger.info(f"Formatting output with accept: {accept}")

    try:
        if accept == 'application/json':
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        logger.error(traceback.format_exc())
        raise