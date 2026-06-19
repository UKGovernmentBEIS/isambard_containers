# TRL Container Changelog

## 1.5.1 / 0.2.7 (2026-06-03)

- Bump TRL to 1.5.1 (from 1.4.0) and rebuild on the `vllm-0.22.1_0.4.0` base
  (rebased from `vllm-0.22.0_0.4.0` to the 0.22.1 patch release; TRL 1.5.1 is
  still the latest release). Build-arg/manifest change only (`sifter.yaml`);
  `trl.def` itself is unchanged.

## 0.2.6 (2026-05-07)

- Mirror the full `vllm.def` `%environment` block in this def. Apptainer wipes
  `/.singularity.d/env/90-environment.sh` from the base before `%setup`/`%post` run,
  so the vllm base's exports are not inherited automatically — each child layer
  must re-declare anything it needs from the base. Includes `LD_LIBRARY_PATH`
  (with `/usr/local/cuda/compat` for cu13 forward compat on driver 565.57.01),
  NCCL/CXI Slingshot config, vLLM usage tag, tiktoken encodings path, JIT compile
  parallelism caps, etc. Keep in sync with `definitions/vllm/vllm.def`.
  Fixes "ProcessGroupNCCL is only supported with GPUs, no GPUs found" / "driver
  too old" errors at NCCL init.

## 0.2.2 (2026-03-23)

- Install `rapidfireai` with `--no-deps` to avoid its stale `huggingface-hub<1.0.0` constraint
  conflicting with `transformers>=5.x` (which needs `huggingface-hub>=1.3.0`).

## 0.2.1 (2026-03-23)

- Pin `transformers` to base image version in constraints to prevent TRL install from downgrading it.

## 0.2.0 (2026-03-23)

- Add `aisi-inspect-tools` (from github.com/AI-Safety-Institute/aisi-inspect-tools)
- Redirect rapidfireai log dir to `$SCRATCH/rapidfireai/logs` (falls back to
  `/tmp`) via `RF_LOG_PATH` env var, avoiding read-only filesystem errors.

## 0.1.0 (2026-03-23)

Initial release. TRL/PEFT training stack on top of vllm-lens base.

Packages installed:

- `trl[peft,deepspeed,liger,quantization,kernels]`
- `datasets>=3.0.0`
- `inspect-ai>=0.3.199`
- `inspect-evals>=0.3.106`
- `openai>=2.29.0`
- `typer>=0.9.0`
- `unsloth>=2026.3.10`
- `rapidfireai>=0.15.2`
- `flash-attn` (built from source for sm_90/aarch64)
