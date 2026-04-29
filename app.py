"""
LLM Infrastructure Planner
- 오픈소스 LLM 모델 선택 → 하드웨어 구성 → 인프라 소요량 자동 계산
- 단일/다중 사용자 시나리오 지원
- NVIDIA/AMD/Intel/NPU 하드웨어 데이터베이스 내장
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math

# ============================================
# 🗄️ 데이터베이스: 모델 & 하드웨어 정의
# ============================================

@dataclass
class LLMModel:
    """오픈소스 LLM 모델 정의"""
    name: str
    variants: Dict[str, float]  # variant_name: param_count_in_billion
    default_variant: str
    architecture: str = "decoder-only"
    moe: bool = False  # Mixture of Experts 여부
    active_params_ratio: float = 1.0  # MoE 모델의 활성 파라미터 비율
    
    def get_params(self, variant: str) -> float:
        """선택된 variant 의 파라미터 수 (Billion) 반환"""
        return self.variants.get(variant, self.variants[self.default_variant])

@dataclass
class GPUHardware:
    """GPU/가속기 하드웨어 정의"""
    name: str
    vendor: str  # "NVIDIA", "AMD", "Intel", "NPU"
    vram_gb: float  # VRAM 용량 (GB)
    bandwidth_gbps: float  # 메모리 대역폭 (GB/s)
    tflops_fp16: float  # FP16 성능 (TFLOPS)
    tflops_int8: float  # INT8 성능 (TOPS)
    max_per_server: int  # 서버당 최대 장착 가능 개수
    server_type: str  # "1U", "2U", "4U", "Workstation"
    nvlink_support: bool = False  # GPU 간 고속 인터커넥트 지원 여부
    price_usd: Optional[float] = None  # 참조용 가격
    
    def get_effective_tops(self, precision: str) -> float:
        """정밀도별 유효 연산 성능 반환"""
        if precision in ["FP16", "BF16"]:
            return self.tflops_fp16 * 1000  # TFLOPS → GFLOPS
        elif precision == "INT8":
            return self.tflops_int8 * 1000
        elif precision == "INT4":
            return self.tflops_int8 * 2000  # INT4 는 INT8 대비 ~2× 효율
        return self.tflops_fp16 * 1000

# ────────────────────────────────────────────
# 🧠 모델 데이터베이스 (오픈소스 중심)
# ────────────────────────────────────────────
MODEL_DB: Dict[str, LLMModel] = {
    "Llama 3": LLMModel(
        name="Llama 3",
        variants={"1B": 1.0, "3B": 3.0, "8B": 8.0, "70B": 70.0, "405B": 405.0},
        default_variant="8B",
        architecture="decoder-only"
    ),
    "Llama 3.1": LLMModel(
        name="Llama 3.1",
        variants={"8B": 8.0, "70B": 70.0, "405B": 405.0},
        default_variant="8B",
        architecture="decoder-only"
    ),
    "Mistral": LLMModel(
        name="Mistral",
        variants={"7B": 7.0},
        default_variant="7B",
        architecture="decoder-only"
    ),
    "Mixtral 8x7B": LLMModel(
        name="Mixtral 8x7B",
        variants={"8x7B": 47.0},  # 총 파라미터
        default_variant="8x7B",
        architecture="MoE",
        moe=True,
        active_params_ratio=0.25  # 활성 파라미터 약 12-14B
    ),
    "Mixtral 8x22B": LLMModel(
        name="Mixtral 8x22B",
        variants={"8x22B": 141.0},
        default_variant="8x22B",
        architecture="MoE",
        moe=True,
        active_params_ratio=0.28
    ),
    "Qwen2.5": LLMModel(
        name="Qwen2.5",
        variants={"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0, "32B": 32.0, "72B": 72.0},
        default_variant="7B",
        architecture="decoder-only"
    ),
    "Gemma 2": LLMModel(
        name="Gemma 2",
        variants={"2B": 2.0, "9B": 9.0, "27B": 27.0},
        default_variant="9B",
        architecture="decoder-only"
    ),
    "Phi-3": LLMModel(
        name="Phi-3",
        variants={"mini-3.8B": 3.8, "small-7B": 7.0, "medium-14B": 14.0},
        default_variant="mini-3.8B",
        architecture="decoder-only"
    ),
    "OLMo": LLMModel(
        name="OLMo",
        variants={"1B": 1.0, "7B": 7.0},
        default_variant="7B",
        architecture="decoder-only"
    ),
    "Falcon": LLMModel(
        name="Falcon",
        variants={"7B": 7.0, "40B": 40.0, "180B": 180.0},
        default_variant="7B",
        architecture="decoder-only"
    ),
}

# ────────────────────────────────────────────
# 💻 GPU 하드웨어 데이터베이스
# ────────────────────────────────────────────
GPU_DB: Dict[str, GPUHardware] = {
    # NVIDIA
    "RTX 4090": GPUHardware(
        name="RTX 4090", vendor="NVIDIA", vram_gb=24, bandwidth_gbps=1008,
        tflops_fp16=330, tflops_int8=660, max_per_server=2,
        server_type="Workstation", nvlink_support=False, price_usd=1600
    ),
    "RTX 6000 Ada": GPUHardware(
        name="RTX 6000 Ada", vendor="NVIDIA", vram_gb=48, bandwidth_gbps=960,
        tflops_fp16=180, tflops_int8=360, max_per_server=4,
        server_type="4U", nvlink_support=True, price_usd=6800
    ),
    "A100 40GB": GPUHardware(
        name="A100 40GB", vendor="NVIDIA", vram_gb=40, bandwidth_gbps=1555,
        tflops_fp16=312, tflops_int8=624, max_per_server=8,
        server_type="4U", nvlink_support=True, price_usd=10000
    ),
    "A100 80GB": GPUHardware(
        name="A100 80GB", vendor="NVIDIA", vram_gb=80, bandwidth_gbps=2039,
        tflops_fp16=312, tflops_int8=624, max_per_server=8,
        server_type="4U", nvlink_support=True, price_usd=15000
    ),
    "H100 80GB": GPUHardware(
        name="H100 80GB", vendor="NVIDIA", vram_gb=80, bandwidth_gbps=3352,
        tflops_fp16=989, tflops_int8=1979, max_per_server=8,
        server_type="4U", nvlink_support=True, price_usd=30000
    ),
    "L40S": GPUHardware(
        name="L40S", vendor="NVIDIA", vram_gb=48, bandwidth_gbps=864,
        tflops_fp16=181, tflops_int8=362, max_per_server=4,
        server_type="2U", nvlink_support=False, price_usd=5500
    ),
    # AMD
    "MI300X": GPUHardware(
        name="MI300X", vendor="AMD", vram_gb=192, bandwidth_gbps=5300,
        tflops_fp16=1300, tflops_int8=2600, max_per_server=8,
        server_type="4U", nvlink_support=True, price_usd=25000
    ),
    "MI250X": GPUHardware(
        name="MI250X", vendor="AMD", vram_gb=128, bandwidth_gbps=3276,
        tflops_fp16=453, tflops_int8=906, max_per_server=8,
        server_type="4U", nvlink_support=True, price_usd=18000
    ),
    # Intel
    "Gaudi 2": GPUHardware(
        name="Gaudi 2", vendor="Intel", vram_gb=96, bandwidth_gbps=2450,
        tflops_fp16=432, tflops_int8=864, max_per_server=8,
        server_type="4U", nvlink_support=True, price_usd=20000
    ),
    # NPU (Copilot+ PC 등)
    "Qualcomm Hexagon": GPUHardware(
        name="Qualcomm Hexagon NPU", vendor="NPU", vram_gb=16, bandwidth_gbps=100,
        tflops_fp16=0, tflops_int8=45, max_per_server=1,
        server_type="Laptop", nvlink_support=False, price_usd=0
    ),
    "Intel NPU (Core Ultra)": GPUHardware(
        name="Intel NPU (Core Ultra)", vendor="NPU", vram_gb=16, bandwidth_gbps=80,
        tflops_fp16=0, tflops_int8=34, max_per_server=1,
        server_type="Laptop", nvlink_support=False, price_usd=0
    ),
}

# ============================================
# 🧮 핵심 계산 엔진
# ============================================

def calculate_model_memory(params_billion: float, precision: str, moe: bool = False, active_ratio: float = 1.0) -> Dict[str, float]:
    """
    모델 메모리 요구량 계산
    반환: {total_gb, weights_gb, moe_active_gb}
    """
    precision_bytes = {"FP32": 4.0, "BF16": 2.0, "FP16": 2.0, "FP8": 1.0, "INT8": 1.0, "INT4": 0.5}
    bytes_per_param = precision_bytes.get(precision, 2.0)
    
    weights_gb = params_billion * bytes_per_param
    
    if moe:
        # MoE 모델: 전체 가중치는 로드해야 하지만, 추론 시 활성 파라미터만 연산
        active_gb = params_billion * active_ratio * bytes_per_param
    else:
        active_gb = weights_gb
    
    # 오버헤드 (옵티마이저 상태, 임시 버퍼 등) - 추론 시 약 10-15%
    overhead_gb = weights_gb * 0.12
    
    return {
        "weights_gb": round(weights_gb, 2),
        "active_gb": round(active_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_gb": round(weights_gb + overhead_gb, 2)
    }

def calculate_kv_cache(params_billion: float, precision: str, context_length: int, 
                       batch_size: int, layers: Optional[int] = None, 
                       hidden_size: Optional[int] = None) -> float:
    """
    KV 캐시 메모리 계산 (GB)
    공식: 2 × layers × hidden_size × context_length × batch_size × precision_bytes / 1GB
    """
    # 파라미터 기반 추정 (레이어/히든 정보가 없을 경우)
    # 일반적인 관계: params ≈ 12 × layers × hidden_size² (대략적)
    # → hidden_size ≈ sqrt(params / (12 × layers))
    
    if layers and hidden_size:
        # 정확한 정보 제공 시
        precision_bytes = {"FP32": 4, "BF16": 2, "FP16": 2, "FP8": 1, "INT8": 1, "INT4": 0.5}
        bytes_per_param = precision_bytes.get(precision, 2)
        kv_gb = (2 * layers * hidden_size * context_length * batch_size * bytes_per_param) / (1024**3)
    else:
        # 추정 모드: 파라미터 수 기반 근사
        # 7B 모델 기준: 레이어 32, 히든 4096 → 토큰당 약 0.5MB (FP16)
        base_kv_per_token_mb = (params_billion / 7.0) * 0.5  # FP16 기준
        precision_factor = {"FP32": 2, "BF16": 1, "FP16": 1, "FP8": 0.5, "INT8": 0.5, "INT4": 0.25}
        kv_per_token_mb = base_kv_per_token_mb * precision_factor.get(precision, 1)
        kv_gb = (kv_per_token_mb * context_length * batch_size) / 1024
    
    return round(kv_gb, 2)

def calculate_inference_memory(params_billion: float, precision: str, context_length: int,
                               batch_size: int, moe: bool = False, active_ratio: float = 1.0,
                               layers: int = None, hidden_size: int = None) -> Dict[str, float]:
    """
    추론 시 총 메모리 요구량 계산
    """
    model_mem = calculate_model_memory(params_billion, precision, moe, active_ratio)
    kv_cache = calculate_kv_cache(params_billion, precision, context_length, batch_size, layers, hidden_size)
    
    # 활성화 메모리 (중간 연산 결과) - 모델 메모리의 15-25%
    activation_gb = model_mem["weights_gb"] * 0.2
    
    total_inference_gb = model_mem["total_gb"] + kv_cache + activation_gb
    
    return {
        "model": model_mem,
        "kv_cache_gb": kv_cache,
        "activation_gb": round(activation_gb, 2),
        "total_gb": round(total_inference_gb, 2)
    }

def calculate_gpu_distribution(total_memory_gb: float, gpu: GPUHardware, precision: str, 
                               allow_tensor_parallel: bool = True) -> Dict:
    """
    단일 모델이 여러 GPU 에 분산되어야 하는지 계산
    반환: {gpus_needed, memory_per_gpu, parallelism_strategy}
    """
    vram_available = gpu.vram_gb * 0.92  # 8% 시스템 오버헤드 제외
    
    if total_memory_gb <= vram_available:
        # 단일 GPU 로 실행 가능
        return {
            "gpus_needed": 1,
            "memory_per_gpu": round(total_memory_gb, 2),
            "strategy": "single-gpu",
            "can_fit": True
        }
    elif allow_tensor_parallel and gpu.nvlink_support:
        # 텐서 병렬화 가능 (NVLink 등 고속 인터커넥트 필요)
        gpus_needed = math.ceil(total_memory_gb / vram_available)
        # 병렬화 오버헤드: 10-20% 추가 통신 버퍼
        overhead_factor = 1.15
        adjusted_per_gpu = (total_memory_gb * overhead_factor) / gpus_needed
        
        if adjusted_per_gpu <= vram_available:
            return {
                "gpus_needed": gpus_needed,
                "memory_per_gpu": round(adjusted_per_gpu, 2),
                "strategy": "tensor-parallel",
                "can_fit": True
            }
    
    # 위 조건 모두 실패 → 모델이 너무 큼
    return {
        "gpus_needed": math.ceil(total_memory_gb / vram_available) + 1,
        "memory_per_gpu": vram_available,
        "strategy": "not-feasible",
        "can_fit": False,
        "shortage_gb": round(total_memory_gb - (vram_available * math.ceil(total_memory_gb / vram_available)), 2)
    }

def estimate_throughput(gpu: GPUHardware, params_billion: float, precision: str, 
                        batch_size: int, context_length: int) -> Dict[str, float]:
    """
    GPU 기반 추론 처리량 추정 (tokens/sec)
    단순화된 경험적 모델 사용
    """
    # 기본 성능: 7B 모델, INT4, 배치 1 기준 ~50 tokens/sec (A100)
    base_tps = 50.0
    
    # 모델 크기 페널티 (파라미터 증가 시 선형 감소)
    size_factor = 7.0 / params_billion
    
    # 정밀도 부스트
    precision_boost = {"INT4": 2.0, "INT8": 1.5, "FP8": 1.2, "FP16": 1.0, "BF16": 1.0, "FP32": 0.5}
    boost = precision_boost.get(precision, 1.0)
    
    # 배치 크기 이점 (선형은 아님, 점점 체감)
    batch_factor = 1 + 0.7 * math.log2(batch_size) if batch_size > 1 else 1.0
    
    # 컨텍스트 길이 페널티 (KV 캐시 증가로 인한 대역폭 병목)
    ctx_factor = 1.0 if context_length <= 8192 else 0.9 ** (math.log2(context_length / 8192))
    
    # 하드웨어 성능 정규화 (A100 FP16 = 312 TFLOPS 기준)
    hw_factor = gpu.get_effective_tops(precision) / (312 * 1000)
    
    tps = base_tps * size_factor * boost * batch_factor * ctx_factor * hw_factor
    
    # 컨텍스트가 매우 길면 메모리 대역폭이 병목
    if context_length > 32768:
        bw_factor = min(1.0, 1500 / gpu.bandwidth_gbps)
        tps *= bw_factor
    
    return {
        "tokens_per_sec": round(max(0.1, tps), 2),
        "time_per_token_ms": round(1000 / max(0.1, tps), 1),
        "batch_throughput": round(max(0.1, tps) * batch_size, 2)
    }

def calculate_infrastructure(model_name: str, variant: str, precision: str, 
                             context_length: int, gpu_name: str, gpus_per_server: int,
                             scenario: str, target_users: int, target_rps: float) -> Dict:
    """
    종합 인프라 계산 메인 함수
    """
    # 1) 모델 정보 로드
    model = MODEL_DB[model_name]
    params_billion = model.get_params(variant)
    
    # 2) 하드웨어 정보 로드
    gpu = GPU_DB[gpu_name]
    
    # 3) 시나리오별 배치 크기 설정
    if scenario == "single-user":
        batch_size = 1
        concurrent_requests = 1
    elif scenario == "multi-user-low":
        batch_size = max(1, target_users // 10)
        concurrent_requests = max(1, target_users // 5)
    else:  # multi-user-high
        batch_size = max(4, target_users // 5)
        concurrent_requests = target_users
    
    # 배치 크기는 하드웨어 한도 내에서 조정
    batch_size = min(batch_size, 32)  # 현실적 상한
    
    # 4) 메모리 계산
    mem_req = calculate_inference_memory(
        params_billion, precision, context_length, batch_size,
        moe=model.moe, active_ratio=model.active_params_ratio
    )
    
    # 5) GPU 분산 필요성 계산
    gpu_dist = calculate_gpu_distribution(mem_req["total_gb"], gpu, precision)
    
    # 6) 처리량 추정
    throughput = estimate_throughput(gpu, params_billion, precision, batch_size, context_length)
    
    # 7) 서버/갯수 계산
    if gpu_dist["can_fit"]:
        gpus_for_model = gpu_dist["gpus_needed"]
        # 서버당 최대 GPU 개수 고려
        servers_needed = math.ceil(gpus_for_model / min(gpus_per_server, gpu.max_per_server))
        actual_gpus_per_server = min(gpus_per_server, gpu.max_per_server, gpus_for_model)
    else:
        servers_needed = -1  # 실행 불가
        actual_gpus_per_server = gpu.max_per_server
        gpus_for_model = gpu_dist["gpus_needed"]
    
    # 8) 다중 사용자 시 추가 리소스 계산
    if scenario != "single-user":
        # 목표 RPS 달성을 위한 복제본 수
        model_tps = throughput["tokens_per_sec"]
        avg_tokens_per_request = 256  # 평균 출력 토큰 수 가정
        requests_per_gpu_per_sec = model_tps / avg_tokens_per_request
        
        if requests_per_gpu_per_sec > 0:
            replicas_needed = math.ceil(target_rps / (requests_per_gpu_per_sec * batch_size))
        else:
            replicas_needed = 999  # 불가능
        
        # 총 인프라 확장
        total_gpus = gpus_for_model * replicas_needed
        total_servers = math.ceil(total_gpus / min(gpus_per_server, gpu.max_per_server))
    else:
        replicas_needed = 1
        total_gpus = gpus_for_model
        total_servers = servers_needed
    
    return {
        "model_info": {
            "name": f"{model_name} {variant}",
            "params_billion": params_billion,
            "precision": precision,
            "context_length": context_length,
            "is_moe": model.moe
        },
        "hardware": {
            "gpu_name": gpu_name,
            "vendor": gpu.vendor,
            "vram_gb": gpu.vram_gb,
            "max_per_server": gpu.max_per_server,
            "server_type": gpu.server_type
        },
        "scenario": {
            "type": scenario,
            "target_users": target_users if scenario != "single-user" else 1,
            "target_rps": target_rps if scenario != "single-user" else 1,
            "batch_size": batch_size,
            "concurrent_requests": concurrent_requests
        },
        "memory": mem_req,
        "gpu_distribution": gpu_dist,
        "throughput": throughput,
        "infrastructure": {
            "gpus_per_model_instance": gpus_for_model,
            "servers_per_instance": servers_needed,
            "replicas_for_throughput": replicas_needed,
            "total_gpus": total_gpus,
            "total_servers": total_servers,
            "gpus_per_server_actual": actual_gpus_per_server
        },
        "feasibility": gpu_dist["can_fit"] and servers_needed > 0
    }

# ============================================
# 🎨 Streamlit UI 구성
# ============================================

def main():
    st.set_page_config(
        page_title="🚀 LLM Infrastructure Planner",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 헤더
    st.title("🚀 LLM Infrastructure Planner")
    st.markdown("""
    *오픈소스 LLM 배포를 위한 하드웨어 인프라 자동 계산기*  
    모델 선택 → 하드웨어 구성 → 인프라 소요량 + 비용 추정까지 한 번에
    """)
    
    # ────────────────────────────────────────
    # 🔧 사이드바: 설정 패널
    # ────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 모델 선택
        st.subheader("🧠 Model Selection")
        model_name = st.selectbox(
            "Select Model",
            options=list(MODEL_DB.keys()),
            index=list(MODEL_DB.keys()).index("Llama 3")
        )
        
        model = MODEL_DB[model_name]
        variant = st.selectbox(
            "Model Variant (Parameter Size)",
            options=list(model.variants.keys()),
            index=list(model.variants.keys()).index(model.default_variant)
        )
        
        precision = st.selectbox(
            "Precision / Quantization",
            options=["INT4", "INT8", "FP8", "FP16", "BF16"],
            index=0  # Default INT4
        )
        
        context_length = st.selectbox(
            "Context Length",
            options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
            index=2  # Default 8K
        )
        
        st.divider()
        
        # 하드웨어 선택
        st.subheader("💻 Hardware Selection")
        
        # 벤더 필터
        vendor_filter = st.selectbox(
            "Hardware Vendor",
            options=["All", "NVIDIA", "AMD", "Intel", "NPU"],
            index=0
        )
        
        # 필터링된 GPU 목록
        gpu_options = [
            name for name, gpu in GPU_DB.items() 
            if vendor_filter == "All" or gpu.vendor == vendor_filter
        ]
        
        gpu_name = st.selectbox(
            "Select GPU/Accelerator",
            options=gpu_options,
            index=gpu_options.index("A100 80GB") if "A100 80GB" in gpu_options else 0
        )
        
        gpu = GPU_DB[gpu_name]
        
        # 서버당 GPU 개수 (하드웨어 최대치 내에서)
        max_gpus = min(gpu.max_per_server, 8)  # 현실적 상한
        gpus_per_server = st.slider(
            "GPUs per Server",
            min_value=1,
            max_value=max_gpus,
            value=min(4, max_gpus)
        )
        
        st.divider()
        
        # 사용 시나리오
        st.subheader("👥 Usage Scenario")
        scenario = st.radio(
            "Deployment Scenario",
            options=["single-user", "multi-user-low", "multi-user-high"],
            format_func=lambda x: {
                "single-user": "🧑 Single User (Local/Dev)",
                "multi-user-low": "👥 Small Team (10-50 users)",
                "multi-user-high": "🏢 Production (100+ users)"
            }[x]
        )
        
        if scenario != "single-user":
            target_users = st.number_input(
                "Expected Concurrent Users",
                min_value=1,
                max_value=10000,
                value=50 if scenario == "multi-user-low" else 200
            )
            target_rps = st.number_input(
                "Target Requests/Second",
                min_value=0.1,
                max_value=1000.0,
                value=5.0 if scenario == "multi-user-low" else 50.0,
                step=0.5
            )
        else:
            target_users = 1
            target_rps = 1.0
        
        # 실행 버튼
        st.divider()
        calculate_btn = st.button("🔍 Calculate Infrastructure", type="primary", use_container_width=True)
    
    # ────────────────────────────────────────
    # 📊 메인 결과 영역
    # ────────────────────────────────────────
    
    # 기본 안내 (계산 전)
    if not calculate_btn:
        st.info("👈 왼쪽 패널에서 모델과 하드웨어를 선택한 후 **Calculate** 버튼을 눌러주세요")
        
        # 미리보기 카드
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Model", f"{model_name} {variant}")
        with col2:
            st.metric("Precision", precision)
        with col3:
            st.metric("Target GPU", gpu_name)
        
        return
    
    # 계산 실행
    with st.spinner("🔄 Calculating infrastructure requirements..."):
        result = calculate_infrastructure(
            model_name=model_name,
            variant=variant,
            precision=precision,
            context_length=context_length,
            gpu_name=gpu_name,
            gpus_per_server=gpus_per_server,
            scenario=scenario,
            target_users=target_users,
            target_rps=target_rps
        )
    
    # ────────────────────────────────────────
    # 📋 결과 표시
    # ────────────────────────────────────────
    
    # 1) 모델 & 하드웨어 요약
    st.subheader("📋 Configuration Summary")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Model", f"{result['model_info']['name']}")
        if result['model_info']['is_moe']:
            st.caption(f"🔀 MoE (Active: {result['model_info']['params_billion'] * model.active_params_ratio:.1f}B)")
    
    with summary_col2:
        st.metric("Precision", result['model_info']['precision'])
        st.metric("Context", f"{result['model_info']['context_length']:,} tokens")
    
    with summary_col3:
        st.metric("GPU", result['hardware']['gpu_name'])
        st.caption(f"{result['hardware']['vendor']} • {result['hardware']['vram_gb']}GB VRAM")
    
    with summary_col4:
        st.metric("Scenario", result['scenario']['type'].replace("-", " ").title())
        if scenario != "single-user":
            st.caption(f"🎯 {target_users} users @ {target_rps} RPS")
    
    st.divider()
    
    # 2) 메모리 분석
    st.subheader("💾 Memory Analysis")
    mem_col1, mem_col2, mem_col3, mem_col4 = st.columns(4)
    
    with mem_col1:
        st.metric("Model Weights", f"{result['memory']['model']['weights_gb']} GB")
    with mem_col2:
        st.metric("KV Cache", f"{result['memory']['kv_cache_gb']} GB")
    with mem_col3:
        st.metric("Activation Buffer", f"{result['memory']['activation_gb']} GB")
    with mem_col4:
        st.metric("🎯 Total Required", f"{result['memory']['total_gb']} GB", 
                  delta=f"{result['hardware']['vram_gb']}GB GPU" if result['memory']['total_gb'] <= result['hardware']['vram_gb'] else None)
    
    # 메모리 분포 차트
    mem_df = pd.DataFrame({
        "Component": ["Model Weights", "KV Cache", "Activation", "Overhead"],
        "Memory (GB)": [
            result['memory']['model']['weights_gb'],
            result['memory']['kv_cache_gb'],
            result['memory']['activation_gb'],
            result['memory']['model']['overhead_gb']
        ]
    })
    
    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        st.bar_chart(mem_df.set_index("Component"), use_container_width=True)
    with chart_col2:
        st.markdown("##### GPU Fit Analysis")
        if result['gpu_distribution']['can_fit']:
            if result['gpu_distribution']['strategy'] == "single-gpu":
                st.success("✅ Fits in single GPU")
            else:
                st.warning(f"⚡ Requires **tensor parallelism** across {result['gpu_distribution']['gpus_needed']} GPUs")
        else:
            st.error(f"❌ Model too large (shortage: {result['gpu_distribution'].get('shortage_gb', 'N/A')} GB)")
    
    st.divider()
    
    # 3) 인프라 요구사항 (핵심)
    st.subheader("🏗️ Infrastructure Requirements")
    
    if not result['feasibility']:
        st.error("⚠️ **Configuration Not Feasible**")
        st.markdown("""
        현재 선택한 모델과 하드웨어 조합으로는 실행이 어렵습니다. 다음을 고려해보세요:
        - 더 높은 양자화 적용 (예: FP16 → INT4)
        - 더 큰 VRAM 을 가진 GPU 선택
        - 더 작은 모델 변형 선택
        - 컨텍스트 길이 단축
        """)
    else:
        infra_col1, infra_col2, infra_col3, infra_col4 = st.columns(4)
        
        with infra_col1:
            st.metric("GPUs per Instance", result['infrastructure']['gpus_per_model_instance'])
            st.caption(f"Strategy: {result['gpu_distribution']['strategy']}")
        
        with infra_col2:
            st.metric("Servers per Instance", result['infrastructure']['servers_per_instance'])
            st.caption(f"{result['hardware']['server_type']} rack")
        
        with infra_col3:
            if scenario != "single-user":
                st.metric("Replicas Needed", result['infrastructure']['replicas_for_throughput'])
                st.caption(f"For {target_rps} RPS target")
            else:
                st.metric("Replicas", "1 (single instance)")
        
        with infra_col4:
            st.metric("🎯 Total GPUs", result['infrastructure']['total_gpus'])
            st.metric("🎯 Total Servers", result['infrastructure']['total_servers'])
        
        # 서버 구성 시각화
        st.markdown("##### Server Configuration Diagram")
        server_viz = f"""
        ```
        {"[Server]" * result['infrastructure']['total_servers']}
           │
           ├─ GPU × {result['infrastructure']['gpus_per_server_actual']} ({result['hardware']['gpu_name']})
           ├─ RAM: ≥ {result['hardware']['vram_gb'] * result['infrastructure']['gpus_per_server_actual'] + 64} GB
           ├─ CPU: ≥ {result['infrastructure']['gpus_per_server_actual'] * 8} cores
           └─ Network: 100GbE+ recommended for multi-GPU
        ```
        """
        st.code(server_viz, language="text")
    
    st.divider()
    
    # 4) 성능 추정
    st.subheader("⚡ Performance Estimate")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Tokens/Second (per GPU)", f"{result['throughput']['tokens_per_sec']}")
    with perf_col2:
        st.metric("Time per Token", f"{result['throughput']['time_per_token_ms']} ms")
    with perf_col3:
        batch_tps = result['throughput']['batch_throughput']
        st.metric(f"Batch Throughput (BS={result['scenario']['batch_size']})", f"{batch_tps} tokens/sec")
    
    # 사용자 경험 변환
    if scenario != "single-user":
        avg_response_tokens = 256
        latency_estimate = (avg_response_tokens / result['throughput']['tokens_per_sec']) * result['infrastructure']['replicas_for_throughput']
        st.caption(f"📊 Estimated P50 Latency: ~{latency_estimate:.1f} seconds per request (at target load)")
    
    st.divider()
    
    # 5) 추천 사항
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    # 메모리 관련
    if not result['gpu_distribution']['can_fit']:
        recommendations.append("🔴 **모델이 너무 큽니다**: INT4 양자화 적용 또는 더 작은 변형 선택을 권장합니다")
    elif result['memory']['total_gb'] > result['hardware']['vram_gb'] * 0.9:
        recommendations.append("🟡 **메모리 여유 부족**: KV 캐시 증가 시 OOM 가능성 있음, 컨텍스트 길이 조정 고려")
    
    # 성능 관련
    if result['throughput']['tokens_per_sec'] < 10:
        recommendations.append("🟡 **성능이 낮음**: 더 높은 TOPS 를 가진 GPU 또는 배치 크기 증가 고려")
    
    # 비용/효율 관련
    if result['infrastructure']['total_gpus'] > 8:
        recommendations.append(f"💰 **대규모 인프라 필요**: {result['infrastructure']['total_gpus']} GPU 는 클라우드 (AWS p4d, Azure NDv5) 검토 권장")
    
    # MoE 모델 특화
    if result['model_info']['is_moe']:
        recommendations.append("🔀 **MoE 모델**: 활성 파라미터만 연산되므로, 추론 시 메모리/성능이 파라미터 수 대비 유리합니다")
    
    # NPU 특화
    if result['hardware']['vendor'] == "NPU":
        if precision not in ["INT4", "INT8"]:
            recommendations.append("⚡ **NPU 최적화**: NPU 는 INT4/INT8 에서 최대 효율을 발휘합니다")
        if result['model_info']['params_billion'] > 10:
            recommendations.append("📱 **NPU 제한**: 10B 이상 모델은 NPU 단독 실행 어려움, 하이브리드 (CPU+GPU) 구성 고려")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.success("✅ Current configuration is well-balanced for your use case!")
    
    # ────────────────────────────────────────
    # 📤 내보내기 기능
    # ────────────────────────────────────────
    st.divider()
    col_exp1, col_exp2 = st.columns([3, 1])
    
    with col_exp1:
        st.caption("💡 계산 결과를 팀과 공유하거나 문서화에 활용하세요")
    
    with col_exp2:
        if st.button("📋 Copy Summary"):
            summary_text = f"""
LLM Infrastructure Plan
=======================
Model: {result['model_info']['name']} ({precision})
Context: {result['model_info']['context_length']} tokens
GPU: {result['hardware']['gpu_name']} × {result['infrastructure']['total_gpus']}
Servers: {result['infrastructure']['total_servers']} ({result['hardware']['server_type']})
Scenario: {result['scenario']['type']}
Throughput: {result['throughput']['tokens_per_sec']} tokens/sec/GPU
Memory: {result['memory']['total_gb']} GB per instance
            """.strip()
            st.code(summary_text, language="text")
            st.toast("📋 Summary copied to clipboard!", icon="✅")

# ============================================
# 🚀 앱 실행
# ============================================

if __name__ == "__main__":
    main()
