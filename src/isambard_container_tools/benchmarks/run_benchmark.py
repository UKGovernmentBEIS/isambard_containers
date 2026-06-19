"""Submit a NCCL networking benchmark to SLURM.

Usage:
    benchmark-networking vllm-lens-0.17.0_0.4.0.sif
    benchmark-networking vllm-lens-0.17.0_0.4.0.sif --backend tcp
    benchmark-networking vllm-lens-0.17.0_0.4.0.sif --nodes 4
    benchmark-networking vllm-lens-0.17.0_0.4.0.sif --reservation interactive
"""

from __future__ import annotations

import logging
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from sifter import resolve_container

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = Path(__file__).resolve().parent / "benchmark_allreduce.py"
TEMPLATES = {
    "slingshot": _PKG_DIR / "templates" / "benchmarks" / "benchmark_slingshot.slurm",
    "tcp": _PKG_DIR / "templates" / "benchmarks" / "benchmark_tcp.slurm",
}

app = typer.Typer(help="Submit a NCCL networking benchmark to SLURM.")


class Backend(str, Enum):
    slingshot = "slingshot"
    tcp = "tcp"


@app.command()
def main(
    container: Annotated[
        str,
        typer.Argument(help="Container name or path (e.g. vllm-lens-0.17.0_0.4.0.sif)"),
    ],
    backend: Annotated[
        Backend,
        typer.Option(
            help="Networking backend: slingshot (native CXI) or tcp (baseline)"
        ),
    ] = Backend.slingshot,
    nodes: Annotated[int, typer.Option(help="Number of nodes")] = 2,
    time: Annotated[str, typer.Option(help="SLURM time limit (HH:MM:SS)")] = "00:10:00",
    partition: Annotated[str, typer.Option(help="SLURM partition")] = "workq",
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
) -> None:
    """Benchmark NCCL all-reduce bandwidth across nodes."""
    if interactive:
        partition = "interactive"
        reservation = "interactive"

    container_path = resolve_container(container)
    if not Path(container_path).exists():
        logger.error("Container not found: %s", container_path)
        raise typer.Exit(1)

    template = TEMPLATES[backend.value]

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)

    env_vars = {
        "CONTAINER": container_path,
        "BENCHMARK_SCRIPT": str(BENCHMARK_SCRIPT),
    }
    export_pairs = ",".join(f"{k}={v}" for k, v in env_vars.items())

    cmd = [
        "sbatch",
        f"--nodes={nodes}",
        f"--time={time}",
        f"--partition={partition}",
        f"--output={logs_dir}/%x_%j.out",
        f"--export=ALL,{export_pairs}",
    ]
    if reservation is not None:
        cmd.append(f"--reservation={reservation}")
    cmd.append(str(template))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("sbatch failed (exit %d): %s", result.returncode, result.stderr)
        raise typer.Exit(result.returncode)

    output = result.stdout.strip()
    match = re.search(r"Submitted batch job (\d+)", output)
    if match:
        job_id = match.group(1)
        typer.echo(f"Submitted {backend.value} benchmark: job {job_id}")
        typer.echo(f"  Container: {container_path}")
        typer.echo(f"  Nodes: {nodes}")
        typer.echo(f"  Logs: {logs_dir}/bench_{backend.value}_{job_id}.out")
    else:
        typer.echo(output)


if __name__ == "__main__":
    app()
