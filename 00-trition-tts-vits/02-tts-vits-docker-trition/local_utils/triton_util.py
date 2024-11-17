import os

account_id_map = {
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "eu-north-1": "601324751636",
    "eu-south-1": "966458181534",
    "eu-central-1": "746233611703",
    "ap-east-1": "110948597952",
    "ap-south-1": "763008648453",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "cn-northwest-1": "474822919863",
    "cn-north-1": "472730292857",
    "sa-east-1": "756306329178",
    "ca-central-1": "464438896020",
    "me-south-1": "836785723513",
    "af-south-1": "774647643957",
}

def setup_triton_client():
    import numpy as np
    import sys

    import tritonclient.grpc as grpcclient

    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001',
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    
    return triton_client, grpcclient

def infer_triton_client(triton_client, model_name, inputs, outputs):
    '''
    Triton 추론 요청
    '''
    # Test with outputs
    # results = triton_client.infer(model_name=model_name,
    #                                 inputs=inputs,
    #                                 outputs=outputs,
    #                                 headers={'test': '1'})


    import time

    # 현재 시간을 밀리초로 변환하여 유니크한 ID 생성
    correlation_id = int(time.time() * 1000)
    request_id = str(correlation_id)  # 문자열로 변환

   #  print("request_id:", request_id)

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        request_id=request_id,
        # sequence_id=1,
        sequence_id=request_id,
        sequence_start=True,  # 시퀀스의 시작을 나타냄
        sequence_end=True,    # 시퀀스의 끝을 나타냄
        headers={'test': '1'}
    )                          
                          
                          
                                                               # Get the output arrays from the results
#     output0_data = results.as_numpy('OUTPUT__0')
#     print("#### output #####")
#     print(output0_data.shape)
#     print("#### output values #####")
#     print(output0_data)    
    
    return results


def make_folder_structure(model_serving_folder, model_name):
    '''
    폴더 구조 생성
    '''
    os.makedirs(model_serving_folder, exist_ok=True)
    os.makedirs(f"{model_serving_folder}/{model_name}", exist_ok=True)
    os.makedirs(f"{model_serving_folder}/{model_name}/1", exist_ok=True)    
    
    return None

def copy_artifact(model_serving_folder, model_name, model_artifact, config):
    '''
    model.pt, config.pbtxt 파일을 지정된 위치에 복사
    '''
    os.system(f"cp {model_artifact} {model_serving_folder}/{model_name}/1/model.pt")    
    os.system(f"cp {config} {model_serving_folder}/{model_name}/config.pbtxt")        
    os.system(f"ls -R {model_serving_folder}")
    
    return None

def remove_folder(model_serving_folder):
    '''
    해당 폴더 전체를 삭제
    '''
    os.system(f"rm -rf  {model_serving_folder}")   
    print(f"{model_serving_folder} is removed") 
    
    return None

def tar_artifact(model_serving_folder, model_name):
    '''
    해당 폴더를 tar 로 압축
    '''
    model_tar_file = f"{model_name}.model.tar.gz"
    os.system(f"tar -C {model_serving_folder}/ -czf {model_tar_file} {model_name}")
    os.system(f"tar tvf {model_tar_file}")
    
    return model_tar_file

def upload_tar_s3(sagemaker_session, tar_file_path, prefix):
    '''
    해당 파일을 S3에 업로딩
    '''
    model_uri_pt = sagemaker_session.upload_data(path=tar_file_path, key_prefix=prefix)
    
    return model_uri_pt

################################
# infernece function
################################


import torch #PyTorch
import commons #VITS 공통함수
import numpy as np
from text import text_to_sequence #텍스트 전처리 함수

#텍스트 전처리 함수
def get_text(text, hps):
    #텍스트를 숫자 시퀀스로 바꿈
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    #blank 토큰 추가 (있는 경우)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    #PyTorch 텐서로 변환    
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def create_input_data(text, hps, noise_scale, noise_scale_w, length_scale):
    #1.텍스트 전처리
    stn_tst = get_text(text, hps) # "hello" --> [12,5,12,3,4,5]
    x_tst = stn_tst #변환된 시퀀스를 x_tst에 할당
    
    #2.텍스트 길이 정보 생성
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda() # 시퀀스 길이를 텐서로 변환하고 GPU로 이동
    
    #3.NumPy 배열로 변환
    x_np = x_tst.detach().cpu().numpy() #GPU > CPU, 텐서 > NumPy
    x_np = x_np.reshape(1,-1) #(N,) > (1,N)형태로 변환
    x_length_np = x_tst_lengths.detach().cpu().numpy() #길이 정보도 NumPy 배열로 변환
    
    #4.생성 파라미터 설정
    #각 파라미터를 float32 타입 NumPy 배열로 변환
    noise_scale_data = np.array([noise_scale], dtype=np.float32) #음성 변화도
    length_scale_data = np.array([noise_scale_w], dtype=np.float32) #발화 속도
    noise_scale_w_data = np.array([length_scale], dtype=np.float32) #음성 다양성

    # (1,N)크기의 텍스트 시퀀스, (1,) 크기의 길이정보, (1,)크기의 음성 변화도 (1,)크기의 발화속도 (1,)크기의 음성 다양성
    return (x_np, x_length_np, noise_scale_data, length_scale_data, noise_scale_w_data)

#1.함수 정의 및 배치 크기 확인
def create_client_payload(x, x_length, noise_scale, length_scale, noise_scale_w, grpcclient):
    inputs = []
    batch_size = x.shape[0]     # Determine the batch size from x
    
#2.텍스트 입력 설정:
    # x input
    x_input = grpcclient.InferInput('x', x.shape, "INT64")
    x_input.set_data_from_numpy(x)
    inputs.append(x_input)

#3.텍스트 길이 입력 설정:
    # x_length input
    x_length_input = grpcclient.InferInput('x_length', [batch_size, 1], "INT64")
    x_length_input.set_data_from_numpy(x_length.reshape(batch_size, 1))
    inputs.append(x_length_input)

#4.생성 파라미터 입력 설정:
    # 음성 변화도
    noise_scale_input = grpcclient.InferInput('noise_scale', [batch_size, 1], "FP32")
    noise_scale_input.set_data_from_numpy(np.full((batch_size, 1), noise_scale, dtype=np.float32))
    inputs.append(noise_scale_input)

    # 발화 속도
    length_scale_input = grpcclient.InferInput('length_scale', [batch_size, 1], "FP32")
    length_scale_input.set_data_from_numpy(np.full((batch_size, 1), length_scale, dtype=np.float32))
    inputs.append(length_scale_input)

    # 음성 다양성
    noise_scale_w_input = grpcclient.InferInput('noise_scale_w', [batch_size, 1], "FP32")
    noise_scale_w_input.set_data_from_numpy(np.full((batch_size, 1), noise_scale_w, dtype=np.float32))
    inputs.append(noise_scale_w_input)
    
#5. 형태 출력:
    print("x data shape:", x.shape)
    print("x_length shape:", x_length.reshape(batch_size, 1).shape)
    print("noise_scale shape:", (batch_size, 1))
    print("length_scale shape:", (batch_size, 1))
    print("noise_scale_w shape:", (batch_size, 1))
    
    return inputs


########################################
#  Create data with sequence_batchhing
########################################


def create_client_payload_sequence_batchhing(x, x_length, noise_scale, length_scale, noise_scale_w, grpcclient):
    inputs = []
    batch_size = x.shape[0]     # Determine the batch size from x
    
#2.텍스트 입력 설정:
    # x input
    x_input = grpcclient.InferInput('x__0', x.shape, "INT64")
    x_input.set_data_from_numpy(x)
    inputs.append(x_input)
    
    print("x_input: ", x_input)

#3.텍스트 길이 입력 설정:
    # x_length input
    x_length_input = grpcclient.InferInput('x_length__1', [batch_size, 1], "INT64")
    x_length_input.set_data_from_numpy(x_length.reshape(batch_size, 1))
    inputs.append(x_length_input)

#4.생성 파라미터 입력 설정:
    # 음성 변화도
    noise_scale_input = grpcclient.InferInput('noise_scale__2', [batch_size, 1], "FP32")
    noise_scale_input.set_data_from_numpy(np.full((batch_size, 1), noise_scale, dtype=np.float32))
    inputs.append(noise_scale_input)

    # 발화 속도
    length_scale_input = grpcclient.InferInput('length_scale__3', [batch_size, 1], "FP32")
    length_scale_input.set_data_from_numpy(np.full((batch_size, 1), length_scale, dtype=np.float32))
    inputs.append(length_scale_input)

    # 음성 다양성
    noise_scale_w_input = grpcclient.InferInput('noise_scale_w__4', [batch_size, 1], "FP32")
    noise_scale_w_input.set_data_from_numpy(np.full((batch_size, 1), noise_scale_w, dtype=np.float32))
    inputs.append(noise_scale_w_input)
    
#5. 형태 출력:
    print("x data shape:", x.shape)
    print("x_length shape:", x_length.reshape(batch_size, 1).shape)
    print("noise_scale shape:", (batch_size, 1))
    print("length_scale shape:", (batch_size, 1))
    print("noise_scale_w shape:", (batch_size, 1))
    
    return inputs


def create_input_payload(text, hps, grpcclient):
    # Create input data for trition client
    input_vars = create_input_data(text, hps, 
                                  noise_scale=.667, 
                                  noise_scale_w=0.8, 
                                  length_scale=1)

    # Get input variables
    x_np, x_length_np, noise_scale_data, length_scale_data, noise_scale_w_data = input_vars

    # Create payload
    inputs = create_client_payload(x=x_np, 
                          x_length=x_length_np, 
                          noise_scale=noise_scale_data, 
                          length_scale=length_scale_data,
                          noise_scale_w=noise_scale_w_data,
                          grpcclient = grpcclient,
                        )
    return inputs


def create_input_payload_sequence_batching(text, hps, grpcclient):
    # Create input data for trition client
    input_vars = create_input_data(text, hps, 
                                  noise_scale=.667, 
                                  noise_scale_w=0.8, 
                                  length_scale=1)

    # Get input variables
    x_np, x_length_np, noise_scale_data, length_scale_data, noise_scale_w_data = input_vars

    # Create payload
    inputs = create_client_payload_sequence_batchhing(x=x_np, 
                          x_length=x_length_np, 
                          noise_scale=noise_scale_data, 
                          length_scale=length_scale_data,
                          noise_scale_w=noise_scale_w_data,
                          grpcclient = grpcclient,
                        )
    return inputs

def create_input_payload_padding(text, hps, grpcclient):
    # Create input data for trition client
    input_vars = create_input_data_padding(text, hps, 
                                  noise_scale=.667, 
                                  noise_scale_w=0.8, 
                                  length_scale=1,
                                  max_sequence_length=2048
                                  )

    # Get input variables
    x_np, x_length_np, noise_scale_data, length_scale_data, noise_scale_w_data = input_vars

    # Create payload
    inputs = create_client_payload(x=x_np, 
                          x_length=x_length_np, 
                          noise_scale=noise_scale_data, 
                          length_scale=length_scale_data,
                          noise_scale_w=noise_scale_w_data,
                          grpcclient = grpcclient,
                        )
    return inputs

########################################
#  Create data with padding
########################################


import numpy as np

def create_input_data_padding(text, hps, noise_scale, noise_scale_w, length_scale, max_sequence_length=2048):
    #1.텍스트 전처리
    stn_tst = get_text(text, hps) # "hello" --> [12,5,12,3,4,5]
    x_tst = stn_tst #변환된 시퀀스를 x_tst에 할당

    #2.텍스트 길이 정보 생성
    original_length = stn_tst.size(0)
    x_tst_lengths = torch.LongTensor([original_length]).cuda() # 시퀀스 길이를 텐서로 변환하고 GPU로 이동

    #3.NumPy 배열로 변환
    x_np = x_tst.detach().cpu().numpy() #GPU > CPU, 텐서 > NumPy
    print("x_np length", len(x_np))
    #4.패딩 처리
    if len(x_np) > max_sequence_length:
        print(f"Warning: Input length {len(x_np)} exceeds maximum length {max_sequence_length}. Truncating...")
        x_np = x_np[:max_sequence_length]
    elif len(x_np) < max_sequence_length:
        # 패딩 추가
        padding_length = max_sequence_length - len(x_np)
        print("padding_length: ", padding_length)
        x_np = np.pad(x_np, (0, padding_length), mode='constant', constant_values=0)

    print("x_np length after padding", len(x_np))        

    x_np = x_np.reshape(1,-1) #(N,) > (1,N)형태로 변환
    # x_length_np = x_tst_lengths.detach().cpu().numpy() #길이 정보는 원래 길이 유지
    # x_length_np = [max_sequence_length].numpy()
    x_length_np = np.array([max_sequence_length], dtype=np.int64) #음성 변화도

    #4.생성 파라미터 설정
    #각 파라미터를 float32 타입 NumPy 배열로 변환
    noise_scale_data = np.array([noise_scale], dtype=np.float32) #음성 변화도
    length_scale_data = np.array([noise_scale_w], dtype=np.float32) #발화 속도
    noise_scale_w_data = np.array([length_scale], dtype=np.float32) #음성 다양성

    # (1,N)크기의 텍스트 시퀀스, (1,) 크기의 길이정보, (1,)크기의 음성 변화도 (1,)크기의 발화속도 (1,)크기의 음성 다양성
    return (x_np, x_length_np, noise_scale_data, length_scale_data, noise_scale_w_data)   


########################################
# Benchmarking Code
########################################

import time
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import pandas as pd

def create_random_input_set(inputs_list: List[List[Any]]) -> List[Any]:
    """입력 세트들 중에서 랜덤하게 하나를 선택"""
    return random.choice(inputs_list)

def run_single_inference(triton_client, model_name: str, inputs_list: List[List[Any]], 
                        outputs: List[Any]) -> tuple:
    """단일 추론 실행 및 시간 측정 (랜덤 입력 사용)"""
    inputs = create_random_input_set(inputs_list)
    start_time = time.perf_counter()
    result = infer_triton_client(triton_client, model_name, inputs, outputs)
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # ms 단위로 변환
    return result, latency, inputs  # 어떤 입력이 사용되었는지 추적

def run_parallel_inference(triton_client, model_name: str, inputs_list: List[List[Any]], 
                         outputs: List[Any], num_requests: int, 
                         max_workers: int) -> tuple[List[float], Dict]:
    """병렬 추론 실행"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_inference, triton_client, model_name, inputs_list, outputs)
            for _ in range(num_requests)
        ]
        results = [future.result() for future in futures]
    
    # 결과 분리 및 입력별 통계 수집
    latencies = [r[1] for r in results]
    input_stats = {}
    
    # 각 입력 종류별 성능 추적
    for i, result in enumerate(results):
        input_set = str(result[2])  # 입력을 문자열로 변환하여 키로 사용
        if input_set not in input_stats:
            input_stats[input_set] = {
                "count": 0,
                "latencies": []
            }
        input_stats[input_set]["count"] += 1
        input_stats[input_set]["latencies"].append(result[1])
    
    return latencies, input_stats

def benchmark_inference(triton_client, model_name: str, inputs_list: List[List[Any]], 
                       outputs: List[Any], num_requests: int = 100, 
                       max_workers: int = 10, warm_up_rounds: int = 5) -> Dict:
    """벤치마크 실행 및 통계 계산"""
    # # 워밍업 실행 
    print("워밍업 실행 중...")
    for _ in range(warm_up_rounds):
        run_single_inference(triton_client, model_name, inputs_list, outputs)
    
    print(f"벤치마크 시작: {num_requests}개 요청 처리")
    start_time = time.perf_counter()
    latencies, input_stats = run_parallel_inference(triton_client, model_name, inputs_list, 
                                                  outputs, num_requests, max_workers)
    total_time = time.perf_counter() - start_time
    
    # 전체 통계 계산
    overall_stats = {
        "총 요청 수": num_requests,
        "총 처리 시간 (초)": total_time,
        "처리량 (requests/sec)": num_requests / total_time,
        "평균 지연시간 (ms)": np.mean(latencies),
        "최소 지연시간 (ms)": np.min(latencies),
        "최대 지연시간 (ms)": np.max(latencies),
        "중간값 지연시간 (ms)": np.median(latencies),
        "표준편차 (ms)": np.std(latencies),
        "P90 지연시간 (ms)": np.percentile(latencies, 90),
        "P95 지연시간 (ms)": np.percentile(latencies, 95),
        "P99 지연시간 (ms)": np.percentile(latencies, 99),
    }
    
    # 입력별 통계 계산
    input_type_stats = {}
    for input_type, stats in input_stats.items():
        input_type_stats[f"Input {input_type}"] = {
            "요청 수": stats["count"],
            "비율 (%)": (stats["count"] / num_requests) * 100,
            "평균 지연시간 (ms)": np.mean(stats["latencies"]),
            "최소 지연시간 (ms)": np.min(stats["latencies"]),
            "최대 지연시간 (ms)": np.max(stats["latencies"]),
            "P95 지연시간 (ms)": np.percentile(stats["latencies"], 95)
        }
    
    return {"overall": overall_stats, "by_input": input_type_stats}

def save_benchmark_results(stats: Dict, output_file: str = "benchmark_results.csv"):
    """벤치마크 결과를 CSV 파일로 저장"""
    # 전체 통계 저장
    overall_df = pd.DataFrame([stats["overall"]])
    overall_df.to_csv(f"overall_{output_file}", index=False)
    
    # 입력별 통계 저장
    input_stats = []
    for input_type, input_stats_dict in stats["by_input"].items():
        row = {"input_type": input_type, **input_stats_dict}
        input_stats.append(row)
    input_df = pd.DataFrame(input_stats)
    input_df.to_csv(f"by_input_{output_file}", index=False)
    
    print(f"벤치마크 결과가 저장되었습니다:")
    print(f"- 전체 통계: overall_{output_file}")
    print(f"- 입력별 통계: by_input_{output_file}")

# 사용 예시
def run_benchmark_example(triton_client, model_name: str, inputs_list: List[List[Any]], 
                         outputs: List[Any]):
    # 벤치마크 설정
    config = {
        "num_requests": 64,  # 총 요청 수
        "max_workers": 8,     # 동시 처리 워커 수
        "warm_up_rounds": 1    # 워밍업 라운드 수
    }
    
    # 벤치마크 실행
    stats = benchmark_inference(triton_client, model_name, inputs_list, outputs, **config)
    
    # 전체 결과 출력
    print("\n=== 전체 벤치마크 결과 ===")
    for key, value in stats["overall"].items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # 입력별 결과 출력
    print("\n=== 입력별 벤치마크 결과 ===")
    for input_type, input_stats in stats["by_input"].items():
        print(f"\n{input_type}:")
        for key, value in input_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 결과 저장
    save_benchmark_results(stats)
    
    return stats