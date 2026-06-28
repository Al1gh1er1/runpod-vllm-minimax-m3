# Use the full CUDA devel image (same base as official RunPod vLLM worker)
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev curl git build-essential ninja-build && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
RUN ldconfig /usr/local/cuda-13.0/compat/

# Build vLLM from main branch (with MiniMax-M3 support)
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system "pyproject_hooks>=1.2.0" && \
    uv pip install --system "wheel>=0.45.0" && \
    uv pip install --system \
        "vllm @ git+https://github.com/vllm-project/vllm.git"

# Install RunPod worker deps and the official image's requirements
FROM runpod/worker-v1-vllm:v2.22.4

# Copy the upgraded vLLM from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin/vllm /usr/local/bin/vllm

# Reinstall flashinfer and DeepGEMM for compatibility
RUN uv pip install --system "flashinfer>=0.2.1.post1" && \
    uv pip install --system \
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
