# vLLM Image Changelog

All notable changes to the vLLM image will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- vLLM 0.23.0 support (`vllm-0.23.0_0.1.0`, `vllm-lens-0.23.0_0.1.0`, `trl-1.6.0_0.1.0`). vLLM 0.23.0 still pins `torch==2.11.0` on CUDA 13.0.2 and keeps `DEEPEP_COMMIT=73b6ea4`, so the `pytorch-2.11.0-cu1302` base is unchanged and the only `vllm.def` edit is the FlashInfer default bump below. New runtime deps in vLLM's `requirements/cuda.txt` (`tilelang`, `apache-tvm-ffi`, `nvidia-cutlass-dsl[cu13]`, `quack-kernels`, `tokenspeed-mla`, `humming-kernels[cu13]`) are pulled in automatically by the wheel install.

### Changed

- Bumped transformers override to 5.12.1 (from 5.10.2) on the stable + lens builds, tracking the latest transformers release. Satisfies vLLM 0.23.0's `transformers >= 4.56.0` (with the `!= 5.0.*–5.5.0` exclusions).
- FlashInfer bumped to 0.6.12 (from 0.6.11.post2) to match vLLM 0.23.0's `requirements/cuda.txt` pin (also updated as the `vllm.def` default).
- TRL bumped to 1.6.0 (from 1.5.1). vllm-lens stays at 1.1.0 (already latest; requires `vllm>=0.16.0`).

### Added

- vLLM 0.22.1 support (`vllm-0.22.1_0.4.0`, `vllm-lens-0.22.1_0.4.0`) — patch release on the 0.22.0 line. Build inputs are identical to 0.22.0 (`torch==2.11.0`, FlashInfer `0.6.11.post2`, DeepEP `73b6ea4`, and the bitsandbytes/timm/runai pins all unchanged in upstream's Dockerfile), so no `vllm.def` changes were needed — a pure `sifter.yaml` version bump (definition version stays `0.4.0`).
- `VLLM_USE_V2_MODEL_RUNNER` build arg (default `none` = leave vLLM's default). Conditionally exported at runtime; lens builds (`vllm-lens-*`) set it to `"0"` in sifter.yaml to pin the v1 model runner, as vllm-lens is not yet compatible with the v2 runner.
- Build the new Rust frontend (`vllm-rs`, introduced in vLLM 0.22) via upstream's `build_rust.sh` (rustup with the toolchain pinned in `rust-toolchain.toml`, plus a pinned `protoc` from `install_protoc.sh`). The binary is baked into the source tree before the wheel build, so `setup.py` ships it as-is and skips an in-wheel cargo build. `VLLM_REQUIRE_RUST_FRONTEND=1` makes the build fail loudly rather than silently falling back to the Python frontend. `setuptools-rust>=1.9.0` is pulled in automatically via `requirements/build/cuda.txt`.
- `unzip` apt dependency (needed by `install_protoc.sh`).

### Changed

- Bumped transformers override to 5.10.2 (from 5.9.0) on the stable + lens builds, tracking the latest transformers release.
- FlashInfer bumped to 0.6.11.post2 (from 0.6.8.post1) to match vLLM 0.22's Dockerfile default.

### Removed

- Interim `vllm-lens-head-0.22.1rc2_0.0.1` head build — superseded now that the stable `v0.22.1` tag is released (the stable `vllm-lens-0.22.1_0.4.0` build covers the same ground).

### Added

- vLLM 0.21.0 support (`vllm-0.21.0_0.3.1`, `vllm-lens-0.21.0_0.3.1`)
- `DEEPGEMM_PYTHON_INTERPRETERS=$(which python3)` exported before the wheel build — vLLM 0.21 builds DeepGEMM's `_C` per-Python and bundles it into the wheel. Upstream provisions one venv per Python in `requires-python`; we point at the system interpreter instead.

### Changed

- Bumped transformers override to 5.9.0 (from 5.8.1)
- KV connectors flow updated for vLLM 0.21: `nixl-cu13` pre-install removed (upstream `kv_connectors.txt` now pins generic `nixl>=1.1.0`); after the requirements install, `nixl-cu${CUDA_MAJOR}` is force-reinstalled `--no-deps` to land the correct `nixl_ep_cpp.so`.

### Added

- vLLM 0.20.0 support (`vllm-0.20.1_0.1.0`, `vllm-lens-0.20.0_0.1.0`) on pytorch-2.11.0-cu129 base (vLLM pins `torch==2.11.0`)
- Runtime libs for multimodal models: `ffmpeg`, `libsm6`, `libxext6`, `libgl1` (Qwen-VL, Whisper, etc.)
- `libnuma-dev` — required by `fastsafetensors>=0.2.2` (vllm-project/vllm#20384)
- Pre-download of FlashInfer TRTLLM BMM headers at build time so MoE JIT works on air-gapped compute nodes (best-effort, skipped on older flashinfer versions)

### Changed

- Build reqs path handles both old (`requirements/build.txt`) and new (`requirements/build/cuda.txt`) layouts (vLLM restructured in v0.20.0)
- DeepEP now built with `TORCH_CUDA_ARCH_LIST=9.0a` (was `9.0`) to enable Hopper wgmma/TMA
- `bitsandbytes` pinned to `>=0.42.0` (arm64 has no wheels below this)
- `runai-model-streamer` bumped to `>=0.15.7` and added `azure` extra to match upstream vLLM Dockerfile

## [0.2.0] - 2026-03-22

### Added

- vLLM 0.18.0 support (`vllm-0.18.0_0.1.0`, `vllm-lens-0.18.0_0.1.0`)

### Changed

- FlashInfer bumped to 0.6.6 (required by vLLM 0.18.0)
- Transformers override bumped to 5.3.0 (from 5.2.0)
- Dropped `fix_mp_multinode_mq_init.patch` — fix merged upstream in v0.18.0

### Notes

- Ray removed as a vLLM default dependency in v0.18.0; still installed explicitly via `ray[serve]>=2.54.0`
- Cascade attention disabled by default in v0.18.0

## [0.1.0] - 2025-02-05

### Added

- Migrated to sifter-based builds
- Template arguments: `VLLM_VERSION`, `VLLM_COMMIT`
- Support for nightly builds via `VLLM_COMMIT` argument

### Notes

- Build requires ~460GB memory (full GH200 node)
- `MAX_JOBS=64` to prevent OOM during compilation
- Inherits Slingshot NCCL support from base image

## [0.0.4]

### Changed

- **Updated to use `SCRATCHDIR` and `PROJECTDIR`** - All scripts now use the correct Slurm environment variables (`$SCRATCHDIR` and `$PROJECTDIR`) instead of the deprecated `$SCRATCH`. Both directories are mounted into containers at their original paths and passed as environment variables.
- **Added validation for required directories** - Scripts now validate that `SCRATCHDIR` and `PROJECTDIR` are set and fail fast with clear error messages if they're missing.
- **Improved `LOCALDIR` handling** - All scripts now check if `$LOCALDIR` exists before using it. If the directory doesn't exist (even when the variable is set), scripts fall back to `/tmp/${AISI_PLATFORM_USER:-$USER}` with a warning.

### Fixed

- Fixed `ray_worker.slurm` crash when `LOCALDIR` was set but the directory didn't exist - the TLS certificate generation was using the raw `$LOCALDIR` variable instead of the validated `$CONTAINER_TMP`.
- Fixed Triton compilation error (`cannot find /usr/lib64/libc_nonshared.a`) - changed bind mount from `/usr/lib64:/host/usr/lib64:ro` to `/usr/lib64:/usr/lib64:ro`. The container (Ubuntu-based) doesn't have `/usr/lib64/`, so mounting the host's directly is safe.
- Fixed Triton compilation error (`cannot find /lib64/libc.so.6`) - added `/lib64:/lib64:ro` bind mount. The host's `/usr/lib64/libc.so` is a linker script that references `/lib64/libc.so.6` by absolute path.

## [0.0.3]

### Added

- Kimi K2 serving example in README
- Documentation for extending containers with `pip install --user`
- New benchmark scripts: `benchmark_slingshot.slurm`, `benchmark_tcp.slurm`

### Changed

- All serving scripts updated to use native Slingshot by default
- `start_ray_head.sh`, `ray_worker.slurm` use container's built-in Slingshot support
- Reorganised troubleshooting into Build-time and Runtime sections
- Added Pipeline Parallelism KeyError troubleshooting (PP must divide layer count evenly)
- Added warning about `~/.local` package shadowing causing "works for me" issues

## [0.0.2]

### Changed

- vLLM 0.14.0 definition updated to use the new CUDA 12.6 base

## [0.0.1]

### Added

- Initial release
- vLLM 0.14.0 (built from source for aarch64/GH200)
- Ray 2.53.0 with serve extension
- transformers, accelerate, datasets, sentencepiece, protobuf
- Fix for cv2/typing stdlib shadow issue
- Multi-node Ray cluster support with mTLS
- Slurm build scripts for Isambard GH200
