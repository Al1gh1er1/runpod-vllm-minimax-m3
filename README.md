# RunPod vLLM Worker — MiniMax-M3

Custom vLLM worker with MiniMax-M3 support for RunPod Serverless.

## Why?

The official RunPod vLLM worker (`v2.22.4`) bundles vLLM 0.20.2, which does **not** support the `MiniMaxM3SparseForConditionalGeneration` architecture. This image upgrades vLLM to the main branch which has MiniMax-M3 support.

## Usage

1. Create a Serverless template from this repo in RunPod Console
2. Set `MODEL_NAME` to `nvidia/MiniMax-M3-NVFP4` (or your model path)
3. Attach a network volume with the model at `/runpod-volume`
4. Set `TENSOR_PARALLEL_SIZE` to match your GPU count
