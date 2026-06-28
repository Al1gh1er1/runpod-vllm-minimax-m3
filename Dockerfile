FROM runpod/worker-v1-vllm:v2.22.4

# Upgrade vLLM to latest nightly build (main branch, supports MiniMax-M3)
# vLLM 0.20.2 does not support MiniMaxM3SparseForConditionalGeneration;
# nightly builds from https://wheels.vllm.ai/nightly include it
RUN uv pip install --system -U \
    "vllm" --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# 🛠 FIX: vLLM nightly moves RequestLogger between modules.
# engine.py handles this via try/except — gracefully falls back to RequestLogger=None
# since it's never actually used (always passed as None to serving classes).
COPY src/engine.py /src/engine.py

# 🔍 Log vLLM version for traceability
RUN python -c "import vllm; print('vLLM version:', vllm.__version__)"

# Environment for MiniMax-M3 NVFP4 on 2x H200/Blackwell
ENV MODEL_NAME="/runpod-volume" \
    BASE_PATH="/runpod-volume" \
    TENSOR_PARALLEL_SIZE="2" \
    GPU_MEMORY_UTILIZATION="0.90" \
    MAX_MODEL_LEN="65536" \
    MAX_NUM_SEQS="256" \
    ENFORCE_EAGER="true" \
    DTYPE="auto" \
    BLOCK_SIZE="16" \
    SWAP_SPACE="4" \
    KV_CACHE_DTYPE="auto" \
    RAW_OPENAI_OUTPUT="true" \
    MAX_CONCURRENCY="30" \
    TRUST_REMOTE_CODE="false" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    VLLM_USE_DEEP_GEMM=0
