# Build stage: compile vLLM from main branch with MiniMax-M3 support
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04 AS vllm-builder

RUN apt-get update -y && \
    apt-get install -y python3-pip curl git ninja-build && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
RUN ldconfig /usr/local/cuda-13.0/compat/

# Install vLLM from main branch (supports MiniMax-M3 architecture)
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system \
        "vllm @ git+https://github.com/vllm-project/vllm.git[flashinfer]"

# Runtime stage: RunPod worker with custom vLLM
FROM nvidia/cuda:13.0.2-runtime-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y python3-pip curl git && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
RUN ldconfig /usr/local/cuda-13.0/compat/

# Copy vLLM from builder
COPY --from=vllm-builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=vllm-builder /usr/local/bin/vllm /usr/local/bin/vllm

# Install DeepGEMM for MoE performance
RUN uv pip install --system \
    git+https://github.com/deepseek-ai/DeepGEMM.git@714dd1a4a980f7937a74343d19a8eba4fe321480 \
    --no-build-isolation

# Clone and install RunPod worker code
RUN git clone https://github.com/runpod-workers/worker-vllm.git /worker-tmp && \
    cp -r /worker-tmp/src /src && \
    cp /worker-tmp/builder/requirements.txt /requirements.txt && \
    rm -rf /worker-tmp

RUN uv pip install --system -r /requirements.txt
RUN chmod +x /src/start.sh

# Set environment for RunPod worker
ENV BASE_PATH="/runpod-volume" \
    MODEL_NAME="" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    VLLM_USE_DEEP_GEMM=0

ENV PYTHONPATH="/:/vllm-workspace"

# Default CMD
CMD ["/bin/bash", "/src/start.sh"]
