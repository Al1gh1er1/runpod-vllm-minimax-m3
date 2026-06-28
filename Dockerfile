FROM runpod/worker-v1-vllm:v2.22.4

# Upgrade vLLM to nightly build (from main branch, supports MiniMax-M3)
# The official vLLM nightly wheels are pre-compiled and include the
# MiniMaxM3SparseForConditionalGeneration architecture
RUN uv pip install --system -U \
    "vllm" --pre \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly && \
    # Reinstall flashinfer for compatibility with upgraded vLLM
    uv pip install --system "flashinfer>=0.2.1.post1"
