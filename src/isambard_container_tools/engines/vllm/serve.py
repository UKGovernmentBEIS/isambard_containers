"""Submit a vLLM serving job to SLURM.

Usage:
    # Auto-configure from model (uses MODEL_CONFIGS lookup)
    vllm-serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
"""

from __future__ import annotations

import logging
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import dotenv
import typer
from rich.console import Console
from rich.table import Table

from sifter import (
    find_latest_container as _sifter_find_latest,
    get_jobs,
    resolve_container,
)
from isambard_container_tools.engines.vllm.recipes import (
    apply_exclusive_defaults,
    dict_to_cli_args,
    get_default_args,
    get_env_vars,
    get_num_nodes_gpus,
    get_vllm_version,
)

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


_PKG_DIR = Path(__file__).resolve().parent.parent.parent
SBATCH_TEMPLATE = _PKG_DIR / "templates" / "vllm" / "serve_vllm_mp.slurm"

# GH200 120GB defaults
GPUS_PER_NODE = 4

app = typer.Typer(help="Submit a vLLM serving job to SLURM.")


def find_latest_container(
    vllm_version: str | None = None, *, vllm_lens: bool = False
) -> str:
    """Find the latest vLLM container in the sifter registry."""
    prefix = "vllm-lens-" if vllm_lens else "vllm-"
    try:
        return _sifter_find_latest(prefix=prefix, version=vllm_version)
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise typer.Exit(1) from exc


def resolve_vllm_container(
    container: str | None, model: str, *, vllm_lens: bool = False
) -> str:
    """Resolve and validate a vLLM container path.

    If *container* is None, auto-detect the latest container for *model*.
    Otherwise resolve the given container name via sifter.
    Exits with an error if the resolved path does not exist.
    """
    if container is None:
        resolved = find_latest_container(get_vllm_version(model), vllm_lens=vllm_lens)
    else:
        resolved = resolve_container(container)

    if not Path(resolved).exists():
        logger.error("Container not found: %s", resolved)
        raise typer.Exit(1)

    return resolved


def get_hf_token() -> str | None:
    """Get HF token from env var or CLI token file (~/.cache/huggingface/token)."""
    if token := os.environ.get("HF_TOKEN"):
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def get_gimlet_url() -> str:
    """Base URL of the gimlet deployment, read from the GIMLET_URL env var.

    The deployment is site-specific, so there is no default — e.g.
    `export GIMLET_URL=https://gimlet.example.org`.
    """
    url = os.environ.get("GIMLET_URL", "").rstrip("/")
    if not url:
        logger.error(
            "GIMLET_URL is not set. Set it to your gimlet deployment's base URL, "
            "e.g.\n  export GIMLET_URL=https://gimlet.example.org"
        )
        raise typer.Exit(1)
    return url


def _gimlet_agent_url(base_url: str) -> str:
    """Websocket agent endpoint derived from the gimlet base URL."""
    ws_base = base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    return f"{ws_base}/agent"


def _default_gimlet_service_name() -> str:
    """Generate a default gimlet service name: {user}-vllm."""
    # AISI_PLATFORM_USER is an AISI-specific override; falls back to $USER elsewhere.
    user = os.environ.get("AISI_PLATFORM_USER") or os.environ.get("USER", "unknown")
    return f"{user}-vllm"


def _generate_gimlet_token(
    service_name: str, kms_key_arn: str, duration: str = "720h"
) -> Path:
    """Generate a gimlet agent token. Requires gimlet CLI.

    Tokens are cached in ~/.gimlet/{service_name}.token and reused if present.
    Returns the path to the token file.
    """
    token_dir = Path.home() / ".gimlet"
    token_dir.mkdir(mode=0o700, exist_ok=True)
    token_file = token_dir / f"{service_name}.token"

    if token_file.exists():
        logger.debug("Reusing existing gimlet token: %s", token_file)
        return token_file

    if not shutil.which("gimlet"):
        logger.error(
            "gimlet CLI not found. Install with:\n"
            "  uv tool install git+https://github.com/UKGovernmentBEIS/gimlet\n"
            "Or provide a pre-generated token with --gimlet-token-file"
        )
        raise typer.Exit(1)

    user = os.environ.get("AISI_PLATFORM_USER") or os.environ.get("USER", "unknown")
    subject = f"isambard-{service_name}-{user}"

    result = subprocess.run(
        [
            "gimlet",
            "jwt",
            "agent",
            "--subject",
            subject,
            "--service",
            service_name,
            "--duration",
            duration,
            "--kms-key-arn",
            kms_key_arn,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error("gimlet token generation failed: %s", result.stderr.strip())
        raise typer.Exit(1)

    token_file.write_text(result.stdout.strip())
    token_file.chmod(0o600)
    logger.debug("Generated gimlet token: %s", token_file)
    return token_file


def _parse_value(value: str) -> str | int | float | bool:
    """Parse a string value to its appropriate Python type."""
    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_extra_args(args: list[str]) -> dict[str, str | int | float | bool]:
    """Parse extra CLI args from typer.Context.args.

    Handles formats:
    - --flag value
    - --flag=value
    - --flag (bare boolean flag)
    """
    result: dict[str, str | int | float | bool] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            raw = arg[2:]
            if "=" in raw:
                # --flag=value format (split before replacing hyphens to preserve value)
                raw_key, value = raw.split("=", 1)
                result[raw_key.replace("-", "_")] = _parse_value(value)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                # --flag value format
                key = raw.replace("-", "_")
                result[key] = _parse_value(args[i + 1])
                i += 1
            else:
                # Bare --flag (boolean True)
                result[raw.replace("-", "_")] = True
        i += 1
    return result


def _format_container_env(env: dict[str, str]) -> str:
    """Format env dict as semicolon-separated KEY=VALUE string for sbatch export."""
    return ";".join(f"{k}={v}" for k, v in env.items())


# Args managed by the SLURM template itself — the MP template hardcodes
# --distributed-executor-backend and sets --nnodes, --node-rank, --master-addr,
# --master-port from SLURM env vars.  These must not appear in VLLM_ARGS to
# avoid duplication.
_TEMPLATE_MANAGED_ARGS = {
    "async_scheduling",
    "distributed_executor_backend",
    "nnodes",
    "node_rank",
    "master_addr",
    "master_port",
}


def _strip_template_managed_args(vllm_args: dict[str, Any]) -> None:
    """Remove vLLM args that are set by the SLURM template."""
    for key in _TEMPLATE_MANAGED_ARGS:
        vllm_args.pop(key, None)


def load_vllm_config(
    model: str,
    overrides: dict[str, Any] | None = None,
    *,
    extra_env: dict[str, str] | None = None,
    exclusive: bool = True,
    gpus_per_node: int = GPUS_PER_NODE,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load vLLM args and env vars from model recipe, with overrides.

    Returns (vllm_args, env_vars) where env_vars is the raw dict.
    """
    vllm_args = get_default_args(model)
    env_vars = get_env_vars(model)
    if extra_env:
        env_vars.update(extra_env)
    if overrides:
        vllm_args.update(overrides)
    vllm_args.setdefault("tensor_parallel_size", 1)
    vllm_args.setdefault("pipeline_parallel_size", 1)
    vllm_args.setdefault("gpu_memory_utilization", 0.85)
    if exclusive:
        apply_exclusive_defaults(vllm_args, gpus_per_node)
    return vllm_args, env_vars


def submit(
    *,
    model: str,
    nodes: int,
    gpus_per_node: int,
    container: str | None,
    slurm_time: str,
    partition: str,
    vllm_args: str,
    container_env: str = "",
    hf_token: str | None = None,
    job_name: str = "vllm-serve",
    user_script: str | None = None,
    script_args: str = "",
    work_dir: str | None = None,
    host: str = "0.0.0.0",
    serve_port: int = 8000,
    master_port: int = 29500,
    exclusive: bool = True,
    switches: int | None = 1,
    reservation: str | None = None,
    gimlet_token_file: str | None = None,
    gimlet_service_name: str | None = None,
    dependency: str | None = None,
    debug: bool = False,
) -> int:
    """Submit a SLURM batch script using sbatch --export and return the job ID."""
    # Build environment variables for export
    env_vars = {
        "MODEL": model,
        "CONTAINER": container,
        "VLLM_ARGS": vllm_args,
        "CONTAINER_ENV": container_env,
        "HF_TOKEN": hf_token or "",
        "WORK_DIR": work_dir or "",
        "USER_SCRIPT": user_script or "",
        "SCRIPT_ARGS": script_args,
        "LISTEN_HOST": host,
        "SERVE_PORT": str(serve_port),
        "MASTER_PORT": str(master_port),
        "DEBUG_MODE": "1" if debug else "0",
    }

    # Gimlet tunneling (optional)
    if gimlet_token_file:
        gimlet_url = get_gimlet_url()
        env_vars["GIMLET_TOKEN_FILE"] = gimlet_token_file
        env_vars["GIMLET_URL"] = gimlet_url
        env_vars["GIMLET_SERVER_URL"] = _gimlet_agent_url(gimlet_url)
        env_vars["SERVICE_NAME"] = gimlet_service_name or _default_gimlet_service_name()

    # Format as KEY=VALUE pairs for --export
    export_pairs = ",".join(f"{k}={v}" for k, v in env_vars.items())
    export_arg = f"ALL,{export_pairs}"

    # Build sbatch command with SLURM options
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--nodes={nodes}",
        f"--gpus-per-node={gpus_per_node}",
        f"--time={slurm_time}",
        f"--partition={partition}",
        f"--export={export_arg}",
    ]
    if exclusive:
        cmd.append("--exclusive")
    if switches is not None:
        cmd.append(f"--switches={switches}")
    if reservation is not None:
        cmd.append(f"--reservation={reservation}")
    if dependency is not None:
        cmd.append(f"--dependency={dependency}")
    if work_dir:
        cmd.append(f"--chdir={work_dir}")
    cmd.append(str(SBATCH_TEMPLATE))

    logger.debug("Submitting: %s ...", " ".join(cmd[:7]))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("sbatch failed (exit %d): %s", result.returncode, result.stderr)
        raise typer.Exit(result.returncode)

    output = result.stdout.strip()
    logger.debug("%s", output)

    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        logger.warning("Could not parse job ID from sbatch output.")
        raise typer.Exit(1)
    return int(match.group(1))


_TERMINAL_STATES = frozenset(
    {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}
)
_POLL_INTERVAL = 5
_TIMEOUT = 1800  # 30 minutes


def wait_for_running(
    job_id: int,
    model: str,
    vllm_args: dict[str, Any],
    serve_port: int,
    host: str = "0.0.0.0",
    gimlet_service_name: str | None = None,
) -> None:
    """Poll SLURM until the job is RUNNING, wait for vLLM health, print URL."""
    console = Console()
    cancelled = False

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal cancelled
        cancelled = True

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        start = time.monotonic()
        last_state = None

        with console.status(
            f"[bold]Waiting for job {job_id} to start...", spinner="dots"
        ) as status:
            while not cancelled:
                elapsed = time.monotonic() - start
                if elapsed > _TIMEOUT:
                    console.print(
                        f"\nTimed out after {_TIMEOUT}s waiting for job {job_id}.",
                        style="red",
                    )
                    console.print(f"Check status: scontrol show job {job_id}")
                    return

                jobs = get_jobs(job_id=str(job_id))
                if not jobs:
                    console.print(f"WARNING: Job {job_id} not found.", style="yellow")
                    return

                job = jobs[0]
                state = job.state

                if state != last_state:
                    console.log(f"Job {job_id}: {state}")
                    last_state = state

                if state in _TERMINAL_STATES:
                    console.print(
                        f"Job {job_id} ended with state: {state}", style="red"
                    )
                    return

                if state == "RUNNING":
                    head_node = job.batch_host or job.node
                    if not head_node:
                        console.print(
                            "WARNING: Could not determine head node.", style="yellow"
                        )
                        return
                    base_url = f"http://{head_node}:{serve_port}/v1"
                    if host == "127.0.0.1":
                        # Can't health-check localhost from login node
                        console.log(
                            f"vLLM starting on {head_node} (localhost — "
                            "health check skipped, check logs for readiness)"
                        )
                    else:
                        status.update(f"[bold]vLLM starting up on {head_node}...")
                        _wait_for_healthy(
                            console, base_url, cancelled_fn=lambda: cancelled
                        )
                    _print_summary(
                        console,
                        job_id,
                        base_url,
                        model,
                        vllm_args,
                        gimlet_service_name=gimlet_service_name,
                    )
                    return

                time.sleep(_POLL_INTERVAL)

        # Ctrl+C path — cancel the job
        console.print(f"\nCancelling job {job_id}...")
        subprocess.run(
            ["scancel", str(job_id)], capture_output=True, text=True, check=False
        )
        console.print(f"Job {job_id} cancelled.")
    finally:
        signal.signal(signal.SIGINT, old_handler)


_HEALTH_TIMEOUT = 1800  # 30 minutes


def _wait_for_healthy(
    console: Console,
    base_url: str,
    *,
    cancelled_fn: Callable[[], bool],
    timeout: int = _HEALTH_TIMEOUT,
) -> bool:
    """Poll the vLLM /health endpoint until it responds 200 or timeout.

    Returns True if healthy, False on timeout or cancellation.
    """
    # base_url is like http://nid001020:8000/v1 → http://nid001020:8000/health
    health_url = base_url.removesuffix("/v1") + "/health"
    start = time.monotonic()

    while not cancelled_fn():
        if time.monotonic() - start > timeout:
            console.print(
                f"\nvLLM health check timed out after {timeout}s. "
                "The job may still be starting — check logs.",
                style="yellow",
            )
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    console.log("[bold green]vLLM is ready!")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(_POLL_INTERVAL)
    return False


def _print_summary(
    console: Console,
    job_id: int,
    base_url: str,
    model: str,
    vllm_args: dict[str, Any],
    gimlet_service_name: str | None = None,
) -> None:
    """Print the connection details and serving configuration."""
    log_file = Path.cwd() / "logs" / f"vllm-serve_{job_id}.log"

    # When gimlet is active, use the gimlet URL for all user-facing instructions
    if gimlet_service_name:
        gimlet_base = f"{get_gimlet_url()}/services/{gimlet_service_name}"
        user_url = f"{gimlet_base}/v1"
        kms_arn = os.environ.get("GIMLET_KMS_KEY_ARN", "<your-kms-key-arn>")
        client_token_cmd = (
            f'gimlet jwt client --subject $(whoami) --services "{gimlet_service_name}" '
            f"--duration 24h --kms-key-arn {kms_arn}"
        )
        curl_test = (
            f"TOKEN=$({client_token_cmd})\n"
            f"curl -s {user_url}/chat/completions \\\n"
            f'  -H "Authorization: Bearer $TOKEN" \\\n'
            f'  -H "Content-Type: application/json" \\\n'
            f'  -d \'{{"model": "{model}", '
            f'"messages": [{{"role": "user", "content": "Hi"}}]}}\' | python3 -m json.tool'
        )
        inspect_export = (
            f"export VLLM_BASE_URL={user_url}\n"
            f"export VLLM_API_KEY=$({client_token_cmd})\n"
            f"inspect eval my_eval.py --model vllm/{model}"
        )
    else:
        user_url = base_url
        curl_test = (
            f"curl -s {user_url}/chat/completions"
            f' -H "Content-Type: application/json"'
            f' -d \'{{"model": "{model}",'
            f' "messages": [{{"role": "user", "content": "Hi"}}]}}\''
            f" | python3 -m json.tool"
        )
        inspect_export = (
            f"export VLLM_BASE_URL={user_url}\n"
            f"inspect eval my_eval.py --model vllm/{model}"
        )

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")

    if gimlet_service_name:
        table.add_row("Gimlet URL", user_url)
        table.add_row("Node URL", f"{base_url} (localhost only)")
    else:
        table.add_row("vLLM URL", user_url)
    table.add_row("Model", model)
    table.add_row("TP", str(vllm_args.get("tensor_parallel_size", 1)))
    table.add_row("PP", str(vllm_args.get("pipeline_parallel_size", 1)))
    if vllm_args.get("max_model_len") is not None:
        table.add_row("Max model len", str(vllm_args["max_model_len"]))
    table.add_row("Log file", str(log_file))
    table.add_row("Cancel with", f"scancel {job_id}")
    table.add_row("Inspect", inspect_export)
    table.add_row("Test with", curl_test)

    console.print()
    console.print(table)
    console.print()


def format_script_args(kwargs: dict[str, object]) -> str:
    """Format a dict as CLI arguments: ``{"model": "foo"}`` → ``--model foo``."""
    parts: list[str] = []
    for key, value in kwargs.items():
        flag = f"--{key}"
        if value is True:
            parts.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            parts.append(flag)
            parts.extend(shlex.quote(str(v)) for v in value)
        else:
            parts.append(f"{flag} {shlex.quote(str(value))}")
    return " ".join(parts)


def submit_job(
    model: str,
    *,
    script: str | Path | None = None,
    script_kwargs: dict[str, object] | None = None,
    work_dir: str | Path | None = None,
    time_minutes: int = 1440,
    partition: str = "workq",
    container: str | None = None,
    vllm_lens: bool = False,
    gpus_per_node: int = GPUS_PER_NODE,
    job_name: str = "vllm-job",
    exclusive: bool = True,
    switches: int | None = 1,
    reservation: str | None = None,
    env: dict[str, str] | None = None,
    host: str = "0.0.0.0",
    gimlet_token_file: str | None = None,
    gimlet_service_name: str | None = None,
    dependency: str | None = None,
    debug: bool = False,
    **vllm_overrides: Any,
) -> tuple[str, dict[str, Any]]:
    """Submit a vLLM job, optionally running a script after vLLM starts.

    Parameters
    ----------
    model:
        HuggingFace model ID to serve.
    script:
        Path to Python script to run after vLLM starts. If None, runs in serve-only mode.
    script_kwargs:
        Dict of CLI arguments passed to *script*.
        ``{"model": "foo/bar", "verbose": True}`` renders as ``--model foo/bar --verbose``.
    work_dir:
        Working directory for the job. Defaults to PROJECT_ROOT.
    time_minutes:
        SLURM time limit in minutes. Default 1440 (24 hours).
    partition:
        SLURM partition. Default "workq".
    container:
        Path to container. If None, auto-detects latest vllm (or vllm-lens if
        vllm_lens=True) from sifter.
    vllm_lens:
        Use vllm-lens container variant (includes activation inspection plugin).
        Default False (plain vllm container).
    gpus_per_node:
        GPUs per node. Default 4 for GH200.
    job_name:
        SLURM job name.
    switches:
        Max leaf switches to span (1 = same switch). None disables the constraint.
    env:
        Extra environment variables to set inside the container.
        Merged with (and overrides) any env vars from the model recipe.
    dependency:
        SLURM dependency specification (e.g., ``"afterok:12345"``).
    **vllm_overrides:
        Override vLLM args (e.g., ``tensor_parallel_size=4``, ``max_model_len=4096``).

    Returns
    -------
    tuple[str, dict[str, Any]]
        (job_id, vllm_args dict)
    """
    vllm_args, env_vars = load_vllm_config(
        model,
        overrides=vllm_overrides or None,
        extra_env=env,
        exclusive=exclusive,
        gpus_per_node=gpus_per_node,
    )
    container_env = _format_container_env(env_vars)

    # Compute SLURM allocation from parallelism args
    num_nodes, gpus = get_num_nodes_gpus(vllm_args, gpus_per_node)

    # The MP template hardcodes the backend; strip it (and other template-managed
    # args) so they don't appear twice in VLLM_ARGS.
    _strip_template_managed_args(vllm_args)

    # Build CLI args string for vLLM
    vllm_args_str = dict_to_cli_args(vllm_args)

    container = resolve_vllm_container(container, model, vllm_lens=vllm_lens)

    # Format time as HH:MM:SS
    hours, remainder = divmod(time_minutes, 60)
    slurm_time = f"{hours:02d}:{remainder:02d}:00"

    # Format script args
    script_args = format_script_args(script_kwargs) if script_kwargs else ""

    # Resolve work_dir
    resolved_work_dir = str(work_dir) if work_dir else str(Path.cwd())

    offset = random.randint(0, 9999)
    serve_port = 8000 + offset
    master_port = 29500 + offset

    (Path(resolved_work_dir) / "logs").mkdir(exist_ok=True)
    job_id = submit(
        model=model,
        nodes=num_nodes,
        gpus_per_node=gpus,
        container=container,
        slurm_time=slurm_time,
        partition=partition,
        vllm_args=vllm_args_str,
        container_env=container_env,
        hf_token=get_hf_token(),
        job_name=job_name,
        user_script=str(script) if script else None,
        script_args=script_args,
        work_dir=resolved_work_dir,
        host=host,
        serve_port=serve_port,
        master_port=master_port,
        exclusive=exclusive,
        switches=switches,
        reservation=reservation,
        gimlet_token_file=gimlet_token_file,
        gimlet_service_name=gimlet_service_name,
        dependency=dependency,
        debug=debug,
    )
    return str(job_id), vllm_args


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
    # SLURM/infrastructure args (explicit)
    time: Annotated[str, typer.Option(help="SLURM time limit")] = "24:00:00",
    partition: Annotated[str, typer.Option(help="SLURM partition")] = "workq",
    container: Annotated[
        str | None,
        typer.Option(
            help="Path to container (default: latest vllm from sifter; use --vllm-lens for vllm-lens variant)"
        ),
    ] = None,
    gpus_per_node: Annotated[
        int, typer.Option(help="GPUs per node (default: 4 for GH200)")
    ] = GPUS_PER_NODE,
    exclusive: Annotated[
        bool,
        typer.Option(
            help="Request exclusive node access (bumps TP to use all GPUs)",
        ),
    ] = True,
    reservation: Annotated[
        str | None,
        typer.Option(help="SLURM reservation name (e.g. 'interactive')"),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            help="Shorthand for --partition=interactive --reservation=interactive",
        ),
    ] = False,
    # Network / Gimlet
    host: Annotated[
        str,
        typer.Option(
            help="Bind address for vLLM. Use 127.0.0.1 for localhost only "
            "(set automatically with --gimlet).",
        ),
    ] = "0.0.0.0",
    gimlet: Annotated[
        bool,
        typer.Option(help="Enable gimlet HTTPS tunnel for external access"),
    ] = False,
    gimlet_kms_key_arn: Annotated[
        str | None,
        typer.Option(
            help="KMS key ARN for gimlet token generation (or set GIMLET_KMS_KEY_ARN env var)",
            envvar="GIMLET_KMS_KEY_ARN",
        ),
    ] = None,
    gimlet_service_name: Annotated[
        str | None,
        typer.Option(help="Gimlet service name (default: {user}-vllm)"),
    ] = None,
    gimlet_token_file: Annotated[
        str | None,
        typer.Option(
            help="Path to pre-generated gimlet agent token (skips generation)"
        ),
    ] = None,
    gimlet_token_duration: Annotated[
        str,
        typer.Option(help="Gimlet agent token duration (e.g. '720h', '24h', '7d')"),
    ] = "720h",
    debug: Annotated[
        bool,
        typer.Option(
            help="Enable debug logging (NCCL, vLLM, libfabric) and per-node GPU/memory monitoring",
        ),
    ] = False,
    vllm_lens: Annotated[
        bool,
        typer.Option(
            "--vllm-lens/--no-vllm-lens",
            help="Use vllm-lens container (includes activation inspection plugin)",
        ),
    ] = False,
) -> None:
    """Submit a vLLM serving job to SLURM.

    Any extra arguments are passed through to vLLM. For example:
        vllm-serve model --tp 4 --pp 2 --enable-prefix-caching
    """
    console = Console()

    if interactive:
        partition = "interactive"
        reservation = "interactive"

    # Gimlet token resolution
    resolved_token_file: str | None = None
    resolved_service_name = gimlet_service_name or _default_gimlet_service_name()
    if gimlet:
        # Gimlet handles external access — bind to localhost
        host = "127.0.0.1"
        # Pre-flight: check gimlet-agent binary exists on login node
        gimlet_agent = Path.home() / ".local" / "bin" / "gimlet-agent"
        if not gimlet_agent.exists():
            logger.error(
                "gimlet-agent not found at %s. Install with:\n"
                "  curl -L -o ~/.local/bin/gimlet-agent "
                "https://github.com/UKGovernmentBEIS/gimlet/releases/latest/download/"
                "gimlet-agent-linux-arm64\n"
                "  chmod +x ~/.local/bin/gimlet-agent",
                gimlet_agent,
            )
            raise typer.Exit(1)

        if gimlet_token_file:
            resolved_token_file = gimlet_token_file
        elif gimlet_kms_key_arn:
            resolved_token_file = str(
                _generate_gimlet_token(
                    resolved_service_name, gimlet_kms_key_arn, gimlet_token_duration
                )
            )
        else:
            logger.error(
                "gimlet requires either --gimlet-kms-key-arn (or GIMLET_KMS_KEY_ARN env) "
                "or --gimlet-token-file"
            )
            raise typer.Exit(1)

    if not gimlet and host != "127.0.0.1":
        console.print(
            f"[yellow]Note:[/yellow] vLLM will bind to [bold]{host}[/bold] "
            "(accessible to the cluster). "
            "Use [bold]--gimlet[/bold] for secure HTTPS access.\n",
        )

    # Load defaults from model recipe + merge extra CLI args
    extra_args = _parse_extra_args(ctx.args)
    vllm_args, env_vars = load_vllm_config(
        model,
        overrides=extra_args,
        exclusive=exclusive,
        gpus_per_node=gpus_per_node,
    )
    container_env = _format_container_env(env_vars)

    # Compute SLURM allocation from parallelism args
    num_nodes, gpus = get_num_nodes_gpus(vllm_args, gpus_per_node)

    # The MP template hardcodes the backend; strip it (and other template-managed
    # args) so they don't appear twice in VLLM_ARGS.
    _strip_template_managed_args(vllm_args)

    # Build CLI args string for vLLM
    vllm_args_str = dict_to_cli_args(vllm_args)

    container = resolve_vllm_container(container, model, vllm_lens=vllm_lens)

    offset = random.randint(0, 9999)
    serve_port = 8000 + offset
    master_port = 29500 + offset

    (Path.cwd() / "logs").mkdir(exist_ok=True)
    job_id = submit(
        model=model,
        nodes=num_nodes,
        gpus_per_node=gpus,
        container=container,
        slurm_time=time,
        partition=partition,
        vllm_args=vllm_args_str,
        container_env=container_env,
        hf_token=get_hf_token(),
        work_dir=str(Path.cwd()),
        host=host,
        serve_port=serve_port,
        master_port=master_port,
        exclusive=exclusive,
        reservation=reservation,
        gimlet_token_file=resolved_token_file,
        gimlet_service_name=resolved_service_name,
        debug=debug,
    )
    wait_for_running(
        job_id,
        model,
        vllm_args,
        serve_port,
        host=host,
        gimlet_service_name=resolved_service_name if gimlet else None,
    )


if __name__ == "__main__":
    app()
