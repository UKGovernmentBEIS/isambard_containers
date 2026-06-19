# SGLang Image Changelog

All notable changes to the SGLang image will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-02-06

### Changed

- Dropped sgl-kernel source compilation (OOMs at ~242 GB RSS during CUTLASS build)
- Uses prebuilt sgl-kernel with triton fallback on GH200 (sm_90)

### Tested

- DeepSeek V3.2 (685B, FP8) on 4 nodes (TP=16) with triton backends

### Notes

- sgl-kernel prebuilt wheel only has sm_100 binaries — won't load on GH200, but SGLang falls back to triton kernels automatically
- FlashInfer prebuilt wheel uses JIT compilation for sm_90 at runtime
- Model-specific flags handled by container layers (e.g. `sglang-deepseek-v3.2`)

## [0.1.0] - 2026-02-05

### Added

- Initial SGLang image based on PyTorch 2.9.1 + CUDA 12.6
- SGLang v0.5.6.post2 (latest stable)
- FlashInfer for optimized attention kernels (prebuilt wheel)
- Slingshot NCCL support inherited from base
- NCCL_CUMEM_ENABLE=0 fix for multi-node

### Notes

- Alternative to vLLM with better DeepSeek V3 support
- Different distributed model: uses torch.distributed directly, no Ray PlacementGroups
- Multi-node via `--dist-init-addr`, `--nnodes`, `--node-rank`
- Supports DP Attention for DeepSeek MLA optimization
- **Limitation**: Prebuilt FlashInfer/DeepGEMM wheels not compiled for GH200 (sm_90)
  - Use model-specific container layers (e.g. `sglang-deepseek-v3.2`) which select triton backends automatically
