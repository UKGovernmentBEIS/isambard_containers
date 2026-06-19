"""Run vLLM on the current node (no SLURM).

Usage:
    # Serve a model locally (e.g. on an interactive salloc node)
    vllm-local meta-llama/Llama-3.3-70B-Instruct

    # Override port
    vllm-local meta-llama/Llama-3.2-1B-Instruct --port 9000

    # Pass extra vLLM args
    vllm-local model --max-model-len 4096 --enable-prefix-caching
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from isambard_container_tools.engines.vllm.recipes import dict_to_cli_args
from isambard_container_tools.engines.vllm.serve import (
    GPUS_PER_NODE,
    _parse_extra_args,
    _strip_template_managed_args,
    _wait_for_healthy,
    get_hf_token,
    load_vllm_config,
    resolve_vllm_container,
)

logger = logging.getLogger(__name__)

app = typer.Typer(help="Run vLLM on the current node (no SLURM).")


def _validate_single_node(
    vllm_args: dict[str, Any], gpus_per_node: int = GPUS_PER_NODE
) -> None:
    """Exit with an error if the config requires more than one node."""
    tp = vllm_args.get("tensor_parallel_size", 1)
    pp = vllm_args.get("pipeline_parallel_size", 1)
    dp = vllm_args.get("data_parallel_size", 1)
    total = tp * pp * dp
    if total > gpus_per_node:
        logger.error(
            "Model requires TP=%d, PP=%d, DP=%d (%d GPUs across %d nodes). "
            "Local mode only supports single-node (max %d GPUs). "
            "Use vllm-serve for multi-node SLURM jobs.",
            tp,
            pp,
            dp,
            total,
            -(-total // gpus_per_node),  # ceil division
            gpus_per_node,
        )
        raise typer.Exit(1)


def _build_bind_mounts(host_tmp: str) -> str:
    """Build the Singularity bind-mount string."""
    scratch = os.environ.get("SCRATCHDIR", str(Path.cwd()))
    project = os.environ.get("PROJECTDIR", str(Path.cwd()))

    parts = [
        f"{scratch}:{scratch}",
        f"{host_tmp}:/tmp",
        "/lib64:/lib64:ro",
        "/usr/lib64:/usr/lib64:ro",
        "/etc/ssl:/etc/ssl:ro",
        "/var/lib/ca-certificates:/var/lib/ca-certificates:ro",
        "/dev:/dev",
    ]
    if project != scratch:
        parts.append(f"{project}:{project}")
    return ",".join(parts)


def _build_bash_script(
    *,
    container: str,
    bind_mounts: str,
    model: str,
    vllm_cmd_args: str,
    debug: bool,
) -> str:
    """Build the bash wrapper that loads modules and execs singularity."""
    debug_envs = ""
    if debug:
        debug_envs = (
            "export NCCL_DEBUG=INFO\n"
            "export VLLM_LOGGING_LEVEL=DEBUG\n"
            "export TORCH_DISTRIBUTED_DEBUG=DETAIL\n"
            "export FI_LOG_LEVEL=info\n"
        )

    return f"""\
set -euo pipefail
unset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR LOCALDIR TMPDIR 2>/dev/null || true
{debug_envs}\
module load brics/apptainer-multi-node 2>/dev/null || true
ulimit -n 65536 2>/dev/null || ulimit -n 16384 2>/dev/null || true
exec singularity exec --nv --pid --contain \
    --bind {shlex.quote(bind_mounts)} \
    --env TRITON_CACHE_DIR="/tmp/.triton" \
    --env FLASHINFER_CACHE_DIR="/tmp/.flashinfer" \
    --env FLASHINFER_DISABLE_VERSION_CHECK=1 \
    --env VLLM_USE_FLASHINFER_MOE_FP8=0 \
    --env TORCH_EXTENSIONS_DIR="/tmp/.torch_ext" \
    --env TORCH_HOME="/tmp/.torch" \
    --env TORCHINDUCTOR_CACHE_DIR="/tmp/.torchinductor" \
    --env DG_JIT_CACHE_DIR="/tmp/deepgemm_jit" \
    {shlex.quote(container)} \
    bash -c 'source /host/adapt.sh 2>/dev/null || true; exec "$@"' _ \
    bash -c '
        # GH200 overrides applied AFTER adapt.sh: it ends in exec "$@", so when
        # sourced it never returns. Passing them as the command adapt.sh execs runs them with its
        # env inherited. (Single-node local mode: no multi-node GDRCopy knob.)
        export NCCL_CROSS_NIC=1
        export LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH"
        exec "$@"
    ' _ vllm serve {shlex.quote(model)} {vllm_cmd_args}
"""


def _print_summary(
    console: Console,
    base_url: str,
    model: str,
    vllm_args: dict[str, Any],
) -> None:
    """Print connection details for the local server."""
    curl_test = (
        f"curl -s {base_url}/chat/completions"
        f' -H "Content-Type: application/json"'
        f' -d \'{{"model": "{model}",'
        f' "messages": [{{"role": "user", "content": "Hi"}}]}}\''
        f" | python3 -m json.tool"
    )
    inspect_export = (
        f"export VLLM_BASE_URL={base_url}\ninspect eval my_eval.py --model vllm/{model}"
    )

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")

    table.add_row("vLLM URL", base_url)
    table.add_row("Model", model)
    table.add_row("TP", str(vllm_args.get("tensor_parallel_size", 1)))
    pp = vllm_args.get("pipeline_parallel_size", 1)
    if pp > 1:
        table.add_row("PP", str(pp))
    if vllm_args.get("max_model_len") is not None:
        table.add_row("Max model len", str(vllm_args["max_model_len"]))
    table.add_row("Stop with", "Ctrl+C")
    table.add_row("Inspect", inspect_export)
    table.add_row("Test with", curl_test)

    console.print()
    console.print(table)
    console.print()


@app.command(
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": True,
        "ignore_unknown_options": True,
    }
)
def main(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="HuggingFace model ID")],
    container: Annotated[
        str | None,
        typer.Option(help="Path to container (default: latest vllm from sifter)"),
    ] = None,
    gpus_per_node: Annotated[
        int, typer.Option(help="GPUs per node (default: 4)")
    ] = GPUS_PER_NODE,
    exclusive: Annotated[
        bool,
        typer.Option(
            help="Bump TP to use all GPUs on the node",
        ),
    ] = True,
    host: Annotated[
        str,
        typer.Option(help="Bind address for vLLM"),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(help="Port for vLLM API server"),
    ] = 8000,
    debug: Annotated[
        bool,
        typer.Option(
            help="Enable debug logging (NCCL, vLLM, libfabric)",
        ),
    ] = False,
    vllm_lens: Annotated[
        bool,
        typer.Option(
            "--vllm-lens/--no-vllm-lens",
            help="Use vllm-lens container variant",
        ),
    ] = False,
) -> None:
    """Serve a model with vLLM on the current node (no SLURM)."""
    console = Console()

    # ── Recipe loading & arg merging ─────────────────────────────────
    vllm_args, container_env = load_vllm_config(
        model,
        overrides=_parse_extra_args(ctx.args),
        exclusive=exclusive,
        gpus_per_node=gpus_per_node,
    )

    # ── Parallelism guard ────────────────────────────────────────────
    _validate_single_node(vllm_args, gpus_per_node)

    _strip_template_managed_args(vllm_args)
    vllm_args_str = dict_to_cli_args(vllm_args)

    # ── Container resolution ─────────────────────────────────────────
    container = resolve_vllm_container(container, model, vllm_lens=vllm_lens)

    hf_token = get_hf_token()

    # ── Build vLLM command args ──────────────────────────────────────
    master_port = 29500
    vllm_cmd_parts = [
        "--distributed-executor-backend mp",
        "--nnodes 1",
        "--node-rank 0",
        "--master-addr 127.0.0.1",
        f"--master-port {master_port}",
        f"--host {shlex.quote(host)}",
        f"--port {port}",
    ]
    if vllm_args_str:
        vllm_cmd_parts.append(vllm_args_str)
    vllm_cmd = " ".join(vllm_cmd_parts)

    # ── Temp dir & bind mounts ───────────────────────────────────────
    host_tmp = f"/tmp/vllm_local_{os.getpid()}"
    os.makedirs(host_tmp, exist_ok=True)
    bind_mounts = _build_bind_mounts(host_tmp)

    # ── Environment ──────────────────────────────────────────────────
    env = os.environ.copy()
    for k, v in container_env.items():
        env[k] = str(v)
    if hf_token:
        env["HF_TOKEN"] = hf_token

    # ── Build and launch ─────────────────────────────────────────────
    script = _build_bash_script(
        container=container,
        bind_mounts=bind_mounts,
        model=model,
        vllm_cmd_args=vllm_cmd,
        debug=debug,
    )

    console.print(f"Starting vLLM locally: [bold]{model}[/bold]")
    console.print(f"Container: {container}")
    console.print(f"Listening on: {host}:{port}")
    console.print()

    proc = subprocess.Popen(["bash", "-c", script], env=env)

    # Forward signals to the child process
    def _forward_signal(signum: int, _frame: object) -> None:
        if proc.poll() is None:
            proc.send_signal(signum)

    old_sigint = signal.signal(signal.SIGINT, _forward_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _forward_signal)

    try:
        # Poll health
        base_url = f"http://localhost:{port}/v1"
        if _wait_for_healthy(
            console, base_url, cancelled_fn=lambda: proc.poll() is not None
        ):
            _print_summary(console, base_url, model, vllm_args)

        # Block until vLLM exits
        proc.wait()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Clean up temp dir
        subprocess.run(["rm", "-rf", host_tmp], capture_output=True, check=False)


if __name__ == "__main__":
    app()
