#!/usr/bin/env python3

import json
import inference

def test_inference():
    print("=" * 60)
    print("Testing SageMaker Inference Functions")
    print("=" * 60)

    print("\n1. Testing model_fn...")
    model = inference.model_fn(".")
    print("   ✓ Model loaded")

    print("\n2. Testing input_fn...")
    input_data = {
        "texts": [
            "김치찌개와 된장찌개는 한국의 대표 음식입니다.",
            "인공지능 기술이 빠르게 발전하고 있습니다.",
            "서울은 대한민국의 수도입니다."
        ]
    }
    processed = inference.input_fn(json.dumps(input_data), 'application/json')
    print(f"   ✓ Processed {len(processed['texts'])} texts")

    print("\n3. Testing predict_fn...")
    result = inference.predict_fn(processed, model)
    print(f"   ✓ Embedding shape: ({result['num_texts']}, {result['embedding_dim']})")

    print("\n4. Testing output_fn...")
    output_json = inference.output_fn(result, 'application/json')
    output_data = json.loads(output_json)
    print(f"   ✓ Output keys: {list(output_data.keys())}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()