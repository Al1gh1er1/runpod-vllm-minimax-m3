FROM runpod/worker-v1-vllm:v2.22.4

# Upgrade vLLM to main branch (adds MiniMax-M3 support: MiniMaxM3SparseForConditionalGeneration)
# vLLM 0.20.2 does not support this architecture; main branch does (PR merged June 2026)
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system --force-reinstall \
        "vllm @ git+https://github.com/vllm-project/vllm.git" && \
    uv pip install --system \
        "flashinfer>=0.2.1.post1"

# Reinstall DeepGEMM for MoE support
RUN uv pip install --system \
    git+https://github.com/deepseek-ai/DeepGEMM.git@714dd1a4a980f7937a74343d19a8eba4fe321480 \
    --no-build-isolation

# Set environment variables optimized for MiniMax-M3 NVFP4
ENV MODEL_NAME="nvidia/minimax-m3-nvfp4" \
    BASE_PATH="/runpod-volume" \
    TENSOR_PARALLEL_SIZE="2" \
    GPU_MEMORY_UTILIZATION="0.90" \
    MAX_MODEL_LEN="65536" \
    ENFORCE_EAGER="true" \
    DTYPE="auto" \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    MAX_NUM_SEQS="256" \
    BLOCK_SIZE="16" \
    SWAP_SPACE="4" \
    KV_CACHE_DTYPE="auto" \
    RAW_OPENAI_OUTPUT="true" \
    MAX_CONCURRENCY="30" \
    OPENAI_RESPONSE_ROLE="assistant" \
    TRUST_REMOTE_CODE="false"
