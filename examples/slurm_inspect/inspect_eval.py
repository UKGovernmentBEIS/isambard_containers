"""Activation extraction demo with optional throughput benchmarking."""

# ruff: noqa: E402

import logging
import os
import time

import dotenv

from isambard_container_tools._helpers.lustre import patch_lustre_streams

dotenv.load_dotenv()
patch_lustre_streams()

for var in ("INSPECT_TELEMETRY", "INSPECT_API_KEY_OVERRIDE", "INSPECT_REQUIRED_HOOKS"):
    os.environ.pop(var, None)

# Silence noisy HTTP-level debug logs from the OpenAI client / httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from typing import Annotated, Any

import typer
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import match
from inspect_ai.solver import Generate, TaskState, generate, solver


@solver
def generate_with_activations(layers: list[int] = [2, 3, 4]):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get the model and call directly so we can get activations
        model = get_model()
        config = GenerateConfig(
            extra_body={"extra_args": {"output_residual_stream": layers}},
        )
        output = await model.generate(state.messages, config=config)

        # Add messages to state
        if output.choices:
            state.messages.append(output.choices[0].message)
        else:
            state.metadata["error"] = "No choices in model output"
            return state

        # Put the activations details in metadata
        if output.metadata is None or "activations" not in output.metadata:
            state.metadata["error"] = (
                "No activations in response metadata. "
                "The model may have failed during generation."
            )
            return state
        else:
            rs = output.metadata["activations"]["residual_stream"]
            state.metadata["activations_shape"] = list(rs.shape)
            state.metadata["activations_dtype"] = str(rs.dtype)
            state.metadata["activations_norms"] = {
                layer: float(rs[i].norm().item()) for i, layer in enumerate(layers)
            }

            # Verify activation shape consistency
            prompt_ids = output.metadata.get("prompt_token_ids", [])
            gen_ids = output.metadata.get("token_ids", [])
            if prompt_ids and gen_ids:
                expected_seq = len(prompt_ids) + len(gen_ids) - 1
                actual_seq = rs.shape[1]
                if actual_seq != expected_seq:
                    raise ValueError(
                        f"Activation shape mismatch: "
                        f"expected seq_len={expected_seq}, got {actual_seq}"
                    )

        return state

    return solve


ROLE_TO_MESSAGE = {
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
}


def ultrachat_record_to_sample(record: dict[str, Any]) -> Sample:
    messages = record["messages"]
    # Drop the last assistant response so the model generates it
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    chat = [ROLE_TO_MESSAGE[m["role"]](content=m["content"]) for m in messages]
    return Sample(input=chat, target="")


@task
def ultrachat_activations(num_samples: int = 1, vanilla: bool = False):
    dataset = hf_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        sample_fields=ultrachat_record_to_sample,
        limit=num_samples,
    )
    return Task(
        dataset=dataset,
        solver=[generate()] if vanilla else [generate_with_activations()],
        scorer=match(),
    )


def main(
    model: Annotated[str, typer.Option("-m", "--model")] = "Qwen/Qwen3-8B",
    num_samples: Annotated[int, typer.Option("-n", "--num-samples")] = 1,
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
):
    start = time.monotonic()

    # VLLM_BASE_URL is read automatically by the vllm-lens Inspect provider.
    # When set (e.g. by the SLURM template), it connects to the running server
    # instead of spawning a local one.
    # Include Slurm job ID in eval metadata when running on compute nodes
    metadata: dict[str, str] = {}
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        metadata["slurm_job_id"] = slurm_job_id

    results = eval(
        ultrachat_activations(num_samples=num_samples, vanilla=vanilla),
        model=f"vllm-lens/{model}",
        max_tokens=8192,
        max_connections=max_connections,
        fail_on_error=True,
        log_buffer=64,
        log_level="debug" if debug else "warning",
        display="plain",
        log_realtime=False,
        metadata=metadata,
    )

    elapsed = time.monotonic() - start

    log = results[0]
    print(f"Status: {log.status}")
    print(f"Total time: {elapsed:.1f}s")

    # Token usage summary
    for model_name, usage in log.stats.model_usage.items():
        print(
            f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out "
            f"({usage.output_tokens / elapsed:.1f} tok/s)"
        )

    # Print activations shape from first sample (only in non-vanilla mode)
    if log.samples and not vanilla:
        first = log.samples[0]
        if first.metadata.get("error"):
            print(f"Error: {first.metadata['error']}")
        print(f"Activations shape: {first.metadata.get('activations_shape')}")


if __name__ == "__main__":
    typer.run(main)
