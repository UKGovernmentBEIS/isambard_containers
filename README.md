# AISI Isambard Containers

Singularity/Apptainer containers, Slurm templates and helper scripts for running distributed workloads on Isambard GH200. The containers ship more recent CUDA & PyTorch than is otherwise available on Isambard, with Slingshot/NCCL networking configured, plus images specialised for inference and training (vLLM, TRL). We also provide tools to make these easy to use — e.g. `vllm-serve` to serve any model (and optionally run a script against it in the same job).

**Quick links:** [Install](#install) · [Build containers](#build-containers) · [Serve a model with vLLM](#serve-a-model-with-vllm) · [Run a script + serve](#run-a-script-against-a-served-model) · [Train](#train) · [Use containers directly](#use-the-containers-directly) · [Development](#development)

## Key Containers

| Container | Description |
|-----------|-------------|
| **pytorch** | Base image, with more recent PyTorch and CUDA than is otherwise available on Isambard, and Slingshot/NCCL networking configured. Everything else builds on this. |
| **vllm** | Adds [vLLM](https://github.com/vllm-project/vllm) for fast, multi-node LLM inference. |
| **vllm-lens** | vLLM plus [vllm-lens](https://github.com/AI-Safety-Institute/vllm-lens/tree/main/vllm_lens) for top-down interpretability work. |
| **trl** | Adds [TRL](https://github.com/huggingface/trl) and [Unsloth](https://github.com/unslothai/unsloth) for training, on top of vLLM. |

See [sifter.yaml](sifter.yaml) for the full list of builds and exact versions.

> [!WARNING]
> If you launch these containers via Isambard's [`/host/adapt.sh`](https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/) script, we recommend setting `NCCL_CROSS_NIC=1` and `FI_HMEM_CUDA_USE_GDRCOPY=0` *after* it runs for multi-node work. This enables more performant networking and avoids GDR-copy instability with the cluster network.

## Install

```bash
uv add git+https://github.com/AI-Safety-Institute/hpc-containers.git
```

This gives you the helper scripts (`vllm-serve`, `benchmark-networking`, …) and the Python API.

## Build Containers

> [!NOTE]
> **AISI users can usually skip this.** The standard containers are already built and published to the shared registry, so `vllm-serve` and friends will resolve them automatically. You only need to build if you're outside AISI, or are adding/modifying a container definition.

Containers are built with [Sifter](https://github.com/UKGovernmentBEIS/sifter), a CLI for building Apptainer containers on HPC with Slurm and S3 registry support. Builds are defined in [sifter.yaml](sifter.yaml).

```bash
# Install sifter
uv tool install git+https://github.com/UKGovernmentBEIS/sifter.git

# Build everything in sifter.yaml (or preview with --dry-run)
sifter build --all

# Build a single container
sifter build vllm-0.22.1_0.4.0

# Pull a pre-built container from the registry (AISI)
sifter pull vllm-0.22.1_0.4.0.sif
```

The helper scripts resolve container names from the sifter registry automatically, so once a container is built or pulled you can refer to models rather than `.sif` paths.

## Use Containers

### Serve a model with vLLM

Serve a vLLM instance from your [login node](https://docs.isambard.ac.uk/user-documentation/guides/login/#working-on-the-login-node) or [VS Code tunnel](https://docs.isambard.ac.uk/user-documentation/guides/vscode/):

```bash
uv run vllm-serve <hf-model-path>
```

Parallelism settings and node count are chosen automatically for any of the 100+ models in [model_recipes.yaml](src/isambard_container_tools/engines/vllm/model_recipes.yaml). For example, this spins up GLM 5 on 5 nodes (PP=5, TP=4):

```bash
uv run vllm-serve zai-org/GLM-5-FP8
```

Extra flags are passed through to vLLM, and a few shorthands are available. For models that don't have a default recipe yet you can find sensible defaults on the [vLLM Recipes website](https://recipes.vllm.ai/), noting that for evaluations-style work we typically recommend the pipeline/tensor parallelism versions.

```bash
# Custom parallelism
uv run vllm-serve deepseek-ai/DeepSeek-R1-0528 --tp 4 --pp 3

# Interactive partition
uv run vllm-serve meta-llama/Llama-3.3-70B-Instruct --interactive

# Any other vLLM flags pass straight through
uv run vllm-serve deepseek-ai/DeepSeek-R1-0528 --enable-prefix-caching
```

> [!WARNING]
> **Security:** By default, `vllm-serve` binds to `0.0.0.0` — your model is accessible to **anyone on the cluster**. Use `--gimlet` for secure HTTPS access (recommended, AISI only), or `--host 127.0.0.1` for localhost only (pair with `submit_job(script=...)` to run code in the same job).

<details>
<summary><b>External Access with Gimlet (Recommended, AISI Only)</b></summary>

[Gimlet](https://github.com/UKGovernmentBEIS/gimlet) provides secure HTTPS access to your model from anywhere — RPv2, your laptop, or other Isambard nodes — without exposing the port to the cluster.

> [!NOTE]
> Gimlet is deployment-specific. Set `GIMLET_URL` to the base URL of your gimlet deployment (e.g. `https://gimlet.example.org`) — `vllm-serve --gimlet` reads it and passes it through to the job. You also need a KMS key for your deployment: set `GIMLET_KMS_KEY_ARN` to its ARN. If you're at AISI and want access to the Core-Tech deployment, reach out to us for the `GIMLET_URL` and key.

```bash
# One-time setup: point at your gimlet deployment and KMS key (add to ~/.bashrc)
export GIMLET_URL=https://gimlet.example.org
export GIMLET_KMS_KEY_ARN=arn:aws:kms:<region>:<account-id>:key/<key-id>

# One-time setup: install gimlet CLI + agent
uv tool install git+https://github.com/UKGovernmentBEIS/gimlet
curl -L -o ~/.local/bin/gimlet-agent \
  https://github.com/UKGovernmentBEIS/gimlet/releases/latest/download/gimlet-agent-linux-arm64
chmod +x ~/.local/bin/gimlet-agent

# Serve with gimlet tunnel
uv run vllm-serve deepseek-ai/DeepSeek-R1-0528 --gimlet
```

When the job starts, you'll see a gimlet URL in the output:

```
vLLM URL     http://nid001020:8123/v1
Gimlet URL   ${GIMLET_URL}/services/<your-username>-vllm/v1
```

To access the model, generate a client token:

```bash
gimlet jwt client --subject $(whoami) --services "$(whoami)-vllm" \
  --duration 24h --kms-key-arn $GIMLET_KMS_KEY_ARN > ~/.gimlet/client.token

curl $GIMLET_URL/services/$(whoami)-vllm/v1/models \
  -H "Authorization: Bearer $(cat ~/.gimlet/client.token)"
```

Additional gimlet options:

```bash
# Custom service name
uv run vllm-serve model --gimlet --gimlet-service-name my-model

# Pre-generated token (skips auto-generation)
uv run vllm-serve model --gimlet --gimlet-token-file ~/.gimlet/my.token
```

> [!NOTE]
> The AISI gimlet deployment is only accessible to Core-Tech team members. Without gimlet, the default `0.0.0.0` binding is used (accessible to the cluster).

</details>

### Run a script against a served model

`submit_job` starts a vLLM server and then runs your script against it in the same Slurm job. Parallelism and node count are chosen automatically based on the model.

```python
from isambard_container_tools.engines.vllm import submit_job

job_id, cfg = submit_job(
    "meta-llama/Llama-3.3-70B-Instruct",
    script="path/to/your_script.py",
    script_kwargs={"num-samples": 100, "output-dir": "/scratch/results"},
    work_dir="path/to/your_project",
)
print(f"Submitted job {job_id}")
```

Your script receives `--model`, `--num-samples` and `--output-dir` as CLI args, plus an `OPENAI_BASE_URL` environment variable pointing to the local vLLM server (OpenAI-compatible). Use any HTTP client, the OpenAI SDK, or Inspect to make requests. See [examples/slurm_inspect/](examples/slurm_inspect/) for a worked Inspect eval.

### Train

The **trl** container ships [TRL](https://github.com/huggingface/trl) and [Unsloth](https://github.com/unslothai/unsloth) on top of vLLM, so you can run SFT/DPO/GRPO training with vLLM-backed generation. Launch a training script the same way as inference — inside the container, on the nodes Slurm allocates.

We don't ship a default training launcher/slurm template.

> **Tip:** Point a coding agent at the vLLM serve template, [src/isambard_container_tools/templates/vllm/serve_vllm_mp.slurm](src/isambard_container_tools/templates/vllm/serve_vllm_mp.slurm), and ask it to adapt it into a training Slurm script for your TRL/Unsloth job. It already handles multi-node setup, container invocation and networking — you mostly need to swap the served command for your training command.

### Use the containers directly

You don't have to use the helper scripts — you can `singularity exec` the `.sif` files yourself and build your own containers on top. Set `CONTAINER` to a built/pulled image (see [sifter.yaml](sifter.yaml) for current versions):

```bash
export CONTAINER=vllm-0.22.1_0.4.0.sif
```

Extend a container by installing extra packages with `pip install --user` — your `~/.local` directory is mounted in by default:

```bash
# Install a package (runs pip inside the container, installs to ~/.local)
singularity exec $CONTAINER pip install --user cowsay

# Use it
singularity exec $CONTAINER python -c "import cowsay; cowsay.cow('It works')"
```

Mount data into the container with `--bind`. Your home directory is mounted by default; for other paths:

```bash
# Mount $SCRATCHDIR (recommended for large files/models)
singularity exec --bind $SCRATCHDIR:$SCRATCHDIR $CONTAINER python train.py

# Mount multiple paths
singularity exec --bind $SCRATCHDIR:$SCRATCHDIR --bind /path/to/data:/data $CONTAINER python train.py
```

See the [Isambard storage docs](https://docs.isambard.ac.uk/user-documentation/information/system-storage/) for available filesystems (`$HOME`, `$SCRATCHDIR`, `$PROJECTDIR`, …) and quotas.

> [!WARNING]
> User-installed packages in `~/.local` persist across jobs and can shadow container packages, causing version mismatches and "works for me" issues between users. If you hit strange import errors, check `pip list --user` and consider `pip uninstall <pkg>` or `rm -rf ~/.local/lib/python3.12`. For shared dependencies, ask Core-Tech to add them to the container definition and rebuild, or open a PR yourself.

## Development

### Benchmark container networking

To verify a container's NCCL networking is working correctly with Slingshot:

```bash
# Benchmark with native Slingshot (expected ~149 GB/s busbw at 1GB)
uv run benchmark-networking vllm-lens-0.22.1_0.4.0.sif

# Benchmark with TCP sockets as a baseline (expected ~13 GB/s)
uv run benchmark-networking vllm-lens-0.22.1_0.4.0.sif --backend tcp

# Use more nodes
uv run benchmark-networking pytorch-2.11.0-cu1302_0.1.0.sif --nodes 4

# Interactive partition
uv run benchmark-networking vllm-lens-0.22.1_0.4.0.sif --interactive
```

Container names are resolved from the sifter registry automatically. Logs are written to `logs/`.
