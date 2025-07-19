import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

def test_single_request(prompt, model_name="/workspace/local-models/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
    """단일 요청 테스트"""
    start_time = time.time()
    
    try:
        # 한글 프롬프트를 위한 헤더 설정
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,  # 10-500 토큰 범위를 위해 500으로 증가
                "temperature": 0.7
            },
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            return {
                "success": True,
                "response_time": response_time,
                "tokens_used": tokens_used,
                "response": result["choices"][0]["message"]["content"]
            }
        else:
            # 에러 응답 내용도 포함
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            
            return {
                "success": False,
                "response_time": response_time,
                "error": f"HTTP {response.status_code}",
                "error_detail": error_detail
            }
            
    except Exception as e:
        return {
            "success": False,
            "response_time": time.time() - start_time,
            "error": str(e)
        }

def check_available_models():
    """사용 가능한 모델 목록을 확인합니다."""
    try:
        response = requests.get("http://localhost:8080/v1/models")
        response.raise_for_status()
        models = response.json()
        
        # 모델 목록 출력
        print("📋 사용 가능한 모델:")
        if "data" in models and models["data"]:
            for model in models["data"]:
                print(f"  - {model['id']}")
            return models["data"]  # data 배열만 반환
        else:
            print("  사용 가능한 모델이 없습니다.")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 모델 확인 중 오류 발생: {e}")
        return []
    except Exception as e:
        print(f"❌ 모델 목록 파싱 중 오류: {e}")
        return []

def run_benchmark(num_requests=20, concurrent_requests=1):
    """벤치마크 실행"""
    print(f"🚀 벤치마크 시작: {num_requests}개 요청, {concurrent_requests}개 동시 실행")
    print("=" * 50)
    
    # 벤치마크 전체 시작 시간 기록
    benchmark_start_time = time.time()
    
    # 먼저 사용 가능한 모델 확인
    available_models = check_available_models()
    if not available_models:
        print("⚠️  사용 가능한 모델을 찾을 수 없습니다.")
        return
    
    # 첫 번째 모델 사용
    model_name = available_models[0]["id"]
    print(f"🎯 사용할 모델: {model_name}")
    
    # 20개의 다양한 한글 테스트 프롬프트들 (10-500 토큰 범위의 답변이 나오도록 설계)
    test_prompts = [
        "바이브 코딩에 대해서 알려줘",
        "안녕하세요! 간단한 인사말을 해주세요.",
        "파리는 어느 나라의 수도인가요?",
        "1+1은 무엇인가요?",
        "재미있는 농담을 하나 해주세요.",
        "파이썬이란 무엇인가요?",
        "인공지능의 미래에 대해 설명해주세요.",
        "한국의 전통 음식 중 김치에 대해 자세히 설명해주세요.",
        "기후 변화가 우리 생활에 미치는 영향은 무엇인가요?",
        "스마트폰의 장단점을 설명해주세요.",
        "독서의 중요성에 대해 말해주세요.",
        "운동이 건강에 미치는 긍정적인 효과들을 나열해주세요.",
        "요리 초보자를 위한 간단한 요리 팁을 알려주세요.",
        "여행을 갈 때 준비해야 할 필수품들을 정리해주세요.",
        "스트레스 관리 방법에 대해 조언해주세요.",
        "친환경 생활을 위한 실천 방법들을 제안해주세요.",
        "효과적인 시간 관리 방법을 알려주세요.",
        "건강한 식습관을 위한 조언을 해주세요.",
        "취미 생활의 중요성과 추천 취미를 소개해주세요.",
        "디지털 디톡스의 필요성과 방법을 설명해주세요."
    ]
    
    results = []
    
    if concurrent_requests == 1:
        # 순차 실행
        for i in range(num_requests):
            prompt = test_prompts[i % len(test_prompts)]
            print(f"요청 {i+1}/{num_requests}: {prompt[:30]}...")
            
            result = test_single_request(prompt, model_name)
            results.append(result)
            
            if result["success"]:
                print(f"  ✅ 성공 - {result['response_time']:.2f}초, {result['tokens_used']} 토큰")
                print(f"     응답: {result['response'][:50]}...")
            else:
                print(f"  ❌ 실패 - {result['error']}")
                if "error_detail" in result:
                    print(f"     상세: {result['error_detail']}")
    
    else:
        # 동시 실행
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            for i in range(num_requests):
                prompt = test_prompts[i % len(test_prompts)]
                futures.append(executor.submit(test_single_request, prompt, model_name))
            
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                print(f"요청 {i+1}/{num_requests} 완료 - {result['response_time']:.2f}초")
    
    # 벤치마크 전체 종료 시간 기록
    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time
    
    # 결과 분석
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 50)
    print("📊 벤치마크 결과")
    print("=" * 50)
    
    print(f"총 요청 수: {len(results)}")
    print(f"성공: {len(successful_results)}")
    print(f"실패: {len(failed_results)}")
    print(f"성공률: {len(successful_results)/len(results)*100:.1f}%")
    
    if successful_results:
        response_times = [r["response_time"] for r in successful_results]
        tokens_used = [r["tokens_used"] for r in successful_results]
        
        print(f"\n⏱️  응답 시간:")
        print(f"  평균: {statistics.mean(response_times):.2f}초")
        print(f"  중간값: {statistics.median(response_times):.2f}초")
        print(f"  최소: {min(response_times):.2f}초")
        print(f"  최대: {max(response_times):.2f}초")
        
        print(f"\n📝 토큰 사용량:")
        print(f"  평균: {statistics.mean(tokens_used):.1f} 토큰")
        print(f"  총합: {sum(tokens_used)} 토큰")
        
        # 기존 처리량 계산 (개별 응답 시간 기반)
        total_response_time = sum(response_times)
        if total_response_time > 0:
            requests_per_second_individual = len(successful_results) / total_response_time
            print(f"\n🚀 개별 응답 시간 기반 처리량: {requests_per_second_individual:.2f} 요청/초")
        
        # 새로운 총 처리량 계산 (전체 벤치마크 시간 기반)
        print(f"\n🚀 총 처리량 지표:")
        print(f"  전체 실행 시간: {total_benchmark_time:.2f}초")
        print(f"  총 처리량: {len(successful_results)/total_benchmark_time:.2f} 요청/초")
        print(f"  토큰 처리량: {sum(tokens_used)/total_benchmark_time:.1f} 토큰/초")
        
        # 효율성 비교
        if total_response_time > 0:
            efficiency_ratio = total_benchmark_time / total_response_time
            print(f"  효율성 비율: {efficiency_ratio:.2f} (1.0에 가까울수록 효율적)")
    
    if failed_results:
        print(f"\n❌ 실패한 요청들:")
        for i, result in enumerate(failed_results):
            print(f"  {i+1}. {result['error']}")
            if "error_detail" in result:
                print(f"     상세: {result['error_detail']}") 

if __name__ == "__main__":
    # 한글 프롬프트 벤치마크 (20개 요청, 순차 실행)
    run_benchmark(num_requests=20, concurrent_requests=1)
    
    print("\n" + "=" * 50)
    
    # 동시 실행 벤치마크 (20개 요청, 3개 동시 실행)
    run_benchmark(num_requests=20, concurrent_requests=4) 