"""Tests for the vLLM recipe engine."""

from __future__ import annotations

from isambard_container_tools.engines.vllm.recipes import (
    apply_exclusive_defaults,
    dict_to_cli_args,
    get_all_model_ids,
    get_default_args,
    get_num_nodes_gpus,
)


def test_all_recipes_resolve_without_error() -> None:
    """Every model recipe can be loaded and resolved (no broken inheritance)."""
    ids = get_all_model_ids()
    assert len(ids) > 50, "Expected 50+ model recipes"
    for model_id in ids:
        args = get_default_args(model_id)
        assert isinstance(args, dict), f"{model_id} failed to resolve"


def test_unknown_model_returns_empty() -> None:
    """Unknown models get empty defaults, not errors."""
    assert get_default_args("totally-fake/nonexistent-model") == {}


def test_exclusive_defaults_fill_node() -> None:
    """On an exclusive 4-GPU node, TP=1 should be bumped to use all GPUs."""
    args = {"tensor_parallel_size": 1, "pipeline_parallel_size": 1}
    apply_exclusive_defaults(args, gpus_per_node=4)
    assert args["tensor_parallel_size"] == 4


def test_exclusive_defaults_respect_expert_parallel() -> None:
    """Expert parallel models should not have TP bumped."""
    args = {"tensor_parallel_size": 1, "enable_expert_parallel": True}
    apply_exclusive_defaults(args, gpus_per_node=4)
    assert args["tensor_parallel_size"] == 1


def test_node_count_from_parallelism() -> None:
    """TP * PP * DP determines total GPUs, divided by gpus_per_node gives nodes."""
    nodes, _ = get_num_nodes_gpus(
        {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 3,
            "data_parallel_size": 2,
        },
        gpus_per_node=4,
    )
    assert nodes == 6  # 4 * 3 * 2 = 24 GPUs / 4 per node


def test_cli_args_roundtrip() -> None:
    """Python dict → vLLM CLI string handles types correctly."""
    result = dict_to_cli_args(
        {
            "tensor_parallel_size": 4,
            "enable_prefix_caching": True,
            "enforce_eager": False,
            "max_model_len": None,
        }
    )
    assert "--tensor-parallel-size 4" in result
    assert "--enable-prefix-caching" in result
    assert "enforce-eager" not in result
    assert "max-model-len" not in result
