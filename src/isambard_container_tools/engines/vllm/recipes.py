"""Load vLLM serving arguments from YAML recipes."""

from __future__ import annotations

import logging
import math
import shlex
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_RECIPES_PATH = Path(__file__).parent / "model_recipes.yaml"


@lru_cache(maxsize=1)
def _load_raw_recipes() -> dict[str, dict[str, Any]]:
    """Load the raw YAML file."""
    with _RECIPES_PATH.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_recipe(
    model_id: str,
    raw: dict[str, dict[str, Any]],
    seen: set[str] | None = None,
) -> dict[str, Any] | None:
    """Resolve a recipe, following base references.

    Returns a dict with optional keys:
    - "vllm_args": dict of vLLM CLI args
    - "env": dict of environment variables
    """
    if model_id not in raw:
        return None

    if seen is None:
        seen = set()
    if model_id in seen:
        raise ValueError(f"Circular base reference detected: {model_id}")
    seen.add(model_id)

    cfg = dict(raw[model_id])
    base_id = cfg.pop("base", None)

    if base_id:
        base_cfg = _resolve_recipe(base_id, raw, seen)
        if base_cfg is None:
            raise ValueError(f"Base model '{base_id}' not found for '{model_id}'")
        # Deep merge: base vllm_args + child vllm_args, base env + child env
        merged_vllm = {**base_cfg.get("vllm_args", {}), **cfg.get("vllm_args", {})}
        merged_env = {**base_cfg.get("env", {}), **cfg.get("env", {})}
        result: dict[str, Any] = {}
        if merged_vllm:
            result["vllm_args"] = merged_vllm
        if merged_env:
            result["env"] = merged_env
        # Child vllm_version overrides base
        vllm_version = cfg.get("vllm_version") or base_cfg.get("vllm_version")
        if vllm_version:
            result["vllm_version"] = vllm_version
        return result

    # No base — return vllm_args, env, and vllm_version if present
    result = {}
    if "vllm_args" in cfg:
        result["vllm_args"] = cfg["vllm_args"]
    if "env" in cfg:
        result["env"] = cfg["env"]
    if "vllm_version" in cfg:
        result["vllm_version"] = cfg["vllm_version"]
    return result


def get_default_args(model_path: str) -> dict[str, Any]:
    """Get all vLLM args for a model from the recipes YAML.

    Returns the resolved vllm_args dict (with inheritance applied).
    Returns an empty dict if the model is not found.
    """
    raw = _load_raw_recipes()
    cfg = _resolve_recipe(model_path, raw)
    if cfg is None:
        return {}
    return dict(cfg.get("vllm_args", {}))


def get_env_vars(model_path: str) -> dict[str, str]:
    """Get environment variables for a model from the recipes YAML.

    Returns the resolved env dict (with inheritance applied).
    Returns an empty dict if the model is not found or has no env vars.
    """
    raw = _load_raw_recipes()
    cfg = _resolve_recipe(model_path, raw)
    if cfg is None:
        return {}
    return {k: str(v) for k, v in cfg.get("env", {}).items()}


def get_vllm_version(model_path: str) -> str | None:
    """Get the vllm_version override for a model from the recipes YAML.

    Returns the resolved vllm_version string, or None if the model uses the
    default container.
    """
    raw = _load_raw_recipes()
    cfg = _resolve_recipe(model_path, raw)
    if cfg is None:
        return None
    return cfg.get("vllm_version")


def apply_exclusive_defaults(
    vllm_args: dict[str, Any],
    gpus_per_node: int = 4,
) -> None:
    """Bump tensor_parallel_size to use all node GPUs in exclusive jobs.

    When a node is allocated exclusively we should use all its GPUs.
    DP already occupies GPUs alongside TP, so only bump TP when
    tp * dp < gpus_per_node.  When expert parallel (EP) is enabled,
    the remaining GPUs are used for EP, so TP is not bumped.
    Modifies *vllm_args* in place.
    """
    if vllm_args.get("enable_expert_parallel"):
        return
    tp = vllm_args.get("tensor_parallel_size", 1)
    pp = vllm_args.get("pipeline_parallel_size", 1)
    dp = vllm_args.get("data_parallel_size", 1)
    # Don't reduce TP that already exceeds or fills the node (multi-node TP)
    if tp * dp >= gpus_per_node:
        return
    # Multiple PP stages can share a node — count how many GPUs are actually used
    stages_per_node = gpus_per_node // (tp * dp)
    gpus_used = min(pp, stages_per_node) * tp * dp
    if gpus_used < gpus_per_node:
        new_tp = gpus_per_node // dp
        logger.debug(
            "Exclusive mode: bumping tensor_parallel_size %d -> %d",
            tp,
            new_tp,
        )
        vllm_args["tensor_parallel_size"] = new_tp


def get_num_nodes_gpus(
    vllm_args: dict[str, Any],
    gpus_per_node: int = 4,
) -> tuple[int, int]:
    """Compute (num_nodes, gpus_per_node) from parallelism args.

    Standard formula: total_gpus = tp * pp * dp

    Expert Parallel (EP) shares GPUs with TP, so it does not multiply.
    """
    tp = vllm_args.get("tensor_parallel_size", 1)
    pp = vllm_args.get("pipeline_parallel_size", 1)
    dp = vllm_args.get("data_parallel_size", 1)

    total_gpus = tp * pp * dp
    num_nodes = max(1, math.ceil(total_gpus / gpus_per_node))

    return (num_nodes, min(total_gpus, gpus_per_node))


def dict_to_cli_args(args: dict[str, Any]) -> str:
    """Convert a dict of vLLM args to CLI string format.

    Handles:
    - Boolean True -> bare flag (--flag)
    - Boolean False -> omitted
    - None -> omitted
    - str/int/float -> --flag value
    - Values containing spaces or special chars -> quoted
    """
    parts: list[str] = []

    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"

        if value is True:
            parts.append(flag)
        elif value is False or value is None:
            continue
        else:
            parts.append(f"{flag} {shlex.quote(str(value))}")

    return " ".join(parts)


def get_all_model_ids() -> list[str]:
    """Return all model IDs that have recipes (excluding hidden bases)."""
    return [k for k in _load_raw_recipes() if not k.startswith("_")]
