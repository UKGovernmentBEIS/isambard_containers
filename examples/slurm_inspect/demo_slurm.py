"""Native sbatch launcher for activation capture jobs with auto-computed serving configs."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from sifter import api

from isambard_container_tools._helpers.pre_download import (
    DatasetSpec,
    pre_download_datasets,
    pre_download_models,
)
from isambard_container_tools.engines.vllm import get_num_nodes_gpus, submit_job
from isambard_container_tools.engines.vllm.serve import find_latest_container
from isambard_container_tools.engines.vllm.serve import _parse_value  # noqa: PLC2701

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

EVAL_SCRIPT = Path(__file__).resolve().parent / "inspect_eval.py"

ULTRACHAT_DATASET: DatasetSpec = {
    "path": "HuggingFaceH4/ultrachat_200k",
    "split": "train_sft",
}


LARGE_MODELS = [
    "moonshotai/Kimi-K2.5",
    "zai-org/GLM-5-FP8",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3.5-397B-A17B-FP8",
    # "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8",
    # "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8",
]

DEFAULT_MODELS = [
    "google/gemma-3-27b-it",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
] + LARGE_MODELS


def parse_job_output(log_file: Path) -> dict[str, str]:
    """Parse structured output from a completed job's log file.

    Returns a dict with keys like "activations_shape", "total_time", "tok_s", etc.
    """
    if not log_file.exists():
        return {"error": f"No output file: {log_file}"}

    content = log_file.read_text(errors="replace")
    result: dict[str, str] = {}

    # Activation shape (from per-sample output)
    shape_match = re.search(r"Activations shape: (\[.*?\])", content)
    if shape_match:
        result["activations_shape"] = shape_match.group(1)

    # Total time
    time_match = re.search(r"Total time: ([\d.]+)s", content)
    if time_match:
        result["total_time"] = time_match.group(1)

    # Throughput
    tok_match = re.search(r"\(([\d.]+) tok/s\)", content)
    if tok_match:
        result["tok_s"] = tok_match.group(1)

    # Error
    err_match = re.search(r"Error: (.+)", content)
    if err_match:
        result["error"] = err_match.group(1)[:80]

    return result


@dataclass
class JobInfo:
    model: str
    vllm_args: dict[str, Any]
    job_id: str
    job_name: str
    log_file: Path


def get_results(jobs: list[JobInfo], console: Console) -> dict[str, dict[str, str]]:
    """Poll SLURM jobs until all complete, printing results as they arrive."""
    results: dict[str, dict[str, str]] = {}
    terminal_states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
    }

    pending = {j.job_id for j in jobs}
    id_to_job = {j.job_id: j for j in jobs}

    while pending:
        states = {j.job_id: j.state for jid in pending for j in api.status(job_id=jid)}
        for job_id in list(pending):
            state = states.get(job_id, "UNKNOWN")
            if any(state.startswith(s) for s in terminal_states):
                job = id_to_job[job_id]
                if state.startswith("COMPLETED"):
                    parsed = parse_job_output(job.log_file)
                    results[job.model] = parsed
                    summary = parsed.get("tok_s", "")
                    if summary:
                        summary = f"{summary} tok/s"
                    elif parsed.get("error"):
                        summary = parsed["error"]
                    else:
                        summary = parsed.get("activations_shape", "no output")
                    console.print(f"  [green]{job.model}[/] (job {job_id}): {summary}")
                else:
                    results[job.model] = {"error": state}
                    console.print(f"  [red]{job.model}[/] (job {job_id}): {state}")
                pending.discard(job_id)
        time.sleep(10)

    # Print final summary table
    console.print()
    table = Table(title="Results")
    table.add_column("Model", style="cyan")
    table.add_column("GPUs", style="magenta")
    table.add_column("Nodes", style="yellow")
    table.add_column("TP", style="yellow")
    table.add_column("PP", style="yellow")
    table.add_column("Time (s)", style="green")
    table.add_column("Tok/s", style="bold green")
    table.add_column("Activations Shape", style="green")
    for job in jobs:
        parsed = results.get(job.model, {})
        tp = job.vllm_args.get("tensor_parallel_size", 1)
        pp = job.vllm_args.get("pipeline_parallel_size", 1)
        dp = job.vllm_args.get("data_parallel_size", 1)
        total_gpus = tp * pp * dp
        num_nodes, _ = get_num_nodes_gpus(job.vllm_args)
        error = parsed.get("error")
        table.add_row(
            job.model,
            str(total_gpus),
            str(num_nodes),
            str(tp),
            str(pp),
            parsed.get("total_time", "N/A"),
            parsed.get("tok_s", "N/A"),
            f"[red]{error}[/red]" if error else parsed.get("activations_shape", "N/A"),
        )
    console.print(table)

    return results


def demo_slurm(
    model: Annotated[
        list[str], typer.Option("-m", "--model", help="Models to evaluate")
    ] = DEFAULT_MODELS,
    num_samples: Annotated[
        int,
        typer.Option(
            "-n", "--num-samples", help="Number of UltraChat samples per model"
        ),
    ] = 2048,
    container: Annotated[
        str | None,
        typer.Option(help="Container .sif path (default: latest stable from sifter)"),
    ] = None,
    switches: Annotated[
        int | None, typer.Option(help="Max leaf switches to span (1 = same switch)")
    ] = 1,
    vllm_arg: Annotated[
        list[str] | None,
        typer.Option(help="vLLM override as KEY=VALUE (repeatable)"),
    ] = None,
    vanilla: Annotated[
        bool,
        typer.Option(
            "--vanilla/--no-vanilla",
            help="Skip activation capture (plain generation only)",
        ),
    ] = False,
    max_connections: Annotated[
        int, typer.Option(help="Max concurrent connections to the model server")
    ] = 512,
    debug: Annotated[bool, typer.Option(help="Enable debug logging")] = False,
) -> None:
    """Submit SLURM jobs to capture activations for each model."""
    console = Console()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Parse --vllm-arg KEY=VALUE pairs into overrides dict
    vllm_overrides: dict[str, str | int | float | bool] = {}
    for item in vllm_arg or []:
        key, _, value = item.partition("=")
        if not _:
            console.print(f"[red]Invalid --vllm-arg (expected KEY=VALUE): {item}[/red]")
            raise typer.Exit(1)
        vllm_overrides[key] = _parse_value(value)

    # Use the plain vLLM container (without vllm-lens) for vanilla runs
    if vanilla and container is None:
        container = find_latest_container()

    # Compute serving config and submit jobs
    jobs: list[JobInfo] = []

    pre_download_datasets([ULTRACHAT_DATASET])

    for m in model:
        pre_download_models([m])
        job_name = f"demo_{m.split('/')[-1][:30]}"

        job_id, cfg = submit_job(
            model=m,
            work_dir=Path(__file__).resolve().parents[2],
            script=EVAL_SCRIPT,
            script_kwargs={
                "model": m,
                "num-samples": num_samples,
                "max-connections": max_connections,
                **({"vanilla": True} if vanilla else {}),
                **({"debug": True} if debug else {}),
            },
            job_name=job_name,
            time_minutes=800,
            container=container,
            switches=switches,
            env={
                "HF_HUB_OFFLINE": "1",
                "HF_ASSETS_CACHE": f"{os.environ['PROJECTDIR']}/huggingface-cache/assets",
                "HF_HUB_CACHE": f"{os.environ['PROJECTDIR']}/huggingface-cache/hub",
                "HF_XET_CACHE": f"{os.environ['PROJECTDIR']}/huggingface-cache/xet",
                "AISITOOLS_DISABLE_HOOKS": "1",
            },
            debug=debug,
            vllm_lens=True,
            **vllm_overrides,  # type: ignore[arg-type]  # CLI str values coerced by submit_job
        )
        log_file = LOG_DIR / f"{job_name}_{job_id}.log"
        jobs.append(JobInfo(m, cfg, job_id, job_name, log_file))

        tp = cfg.get("tensor_parallel_size", 1)
        pp = cfg.get("pipeline_parallel_size", 1)
        num_nodes, _ = get_num_nodes_gpus(cfg)
        console.print(
            f"  Submitted [cyan]{m}[/cyan]: job {job_id} "
            f"(TP={tp}, PP={pp}, {num_nodes} node(s))"
        )

    get_results(jobs, console)


app = typer.Typer()
app.command()(demo_slurm)

if __name__ == "__main__":
    app()
