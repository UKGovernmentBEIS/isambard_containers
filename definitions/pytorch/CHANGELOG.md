# PyTorch Base Image Changelog

All notable changes to the PyTorch base image will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2025-02-05

### Added

- Migrated to sifter-based builds
- Template arguments: `PYTORCH_VERSION`, `CUDA_VERSION`

### Notes

- The Docker base image and `%setup` CUDA paths are hardcoded
- To change CUDA version, update both the `From:` line and `%setup` paths

## [0.0.4]

### Fixed

- Fixed Singularity build failure when `%setup` scripts (like aws-ofi-nccl's configure) try to create temp files - build scripts now export `TMPDIR` to the local build directory so `config.guess` and similar tools can write temp files.
- Fixed Singularity build "permission denied" error on Phase 1 Lustre - fakeroot uid mapping doesn't work with Phase 1's Lustre configuration. Build scripts now copy all inputs to local tmpfs (`$LOCALDIR` or `/tmp`), build there, then copy the resulting SIF to Lustre.

## [0.0.3]

### Added

- **Native Slingshot NCCL support** - 11x faster inter-node GPU communication
  - Benchmarked: 149 GB/s with native Slingshot vs 13 GB/s with TCP sockets
  - Uses aws-ofi-nccl plugin with CXI provider for HPE Slingshot fabric

### Changed

- **aws-ofi-nccl and dependencies built into base image** - eliminates fragile spack path dependencies. Only Cray libfabric and libcxi need to be mounted from host.
- **ptxas 12.8.93 patch moved to base image** - benefits all app images that compile CUDA code (previously only in vLLM definition)
- Simplified Slingshot bind mounts: only `/opt/cray/libfabric` and `/usr/lib64` required

## [0.0.2]

### Fixed

- Downgraded from CUDA 12.8 to CUDA 12.6 to fix `cudaErrorUnsupportedPtxVersion` error when loading quantized models (e.g., Kimi K2) with Marlin kernels. The Isambard GH200 driver (565.57.01) supports up to CUDA 12.7. Note: driver versions on HPC systems lag behind latest releases as they are set by suppliers for QA reasons.

### Changed

- Base image now uses `nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` (CUDA 12.7 images not published on NGC)
- PyTorch installed from cu126 wheel

### Removed

- `pytorch-2.9.1-cu128.def` - incompatible with current driver

### Known Limitations

- QuTLASS quantization kernels are disabled (requires CUDA 12.8+)

### Build Workarounds

- Patched **ptxas only** to 12.8.93 (keeping cicc at 12.6) to fix nvcc segfault when compiling Flash Attention 3 while maintaining driver compatibility. cicc 12.6 generates PTX compatible with driver 565.57.01; ptxas 12.8 assembles without segfaulting.
  See [vllm#19095](https://github.com/vllm-project/vllm/pull/19095) and [flash-attention#1453](https://github.com/Dao-AILab/flash-attention/issues/1453)

## [0.0.1]

### Added

- Initial release
- CUDA 12.8 + PyTorch 2.9.1 (later downgraded to 12.6 in 0.0.2)
- Python 3.12
- Slurm build scripts for Isambard GH200
