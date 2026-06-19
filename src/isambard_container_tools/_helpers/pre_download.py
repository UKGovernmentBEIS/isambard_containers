"""Pre-download HuggingFace models and datasets to cache."""

from __future__ import annotations

import logging
import os
from typing import NotRequired, TypedDict

logger = logging.getLogger(__name__)


class DatasetSpec(TypedDict):
    """Specification for a HuggingFace dataset to pre-download."""

    path: str
    split: NotRequired[str]


def pre_download_models(models: list[str]) -> None:
    """Download model weights to the HF cache before submitting SLURM jobs.

    Avoids having lots of jobs all trying to download the same model at once,
    which can cause timeouts and failures.
    """
    cache_dir = os.environ.get("HF_HUB_CACHE")
    logger.info(
        "Pre-downloading %d model(s) to %s...",
        len(models),
        cache_dir or "default cache",
    )
    for model_id in models:
        print("Pre-downloading %s..." % model_id)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(model_id, cache_dir=cache_dir)
            logger.info("  %s: OK", model_id)
        except Exception:
            logger.exception("  %s: download failed", model_id)


def pre_download_datasets(datasets: list[DatasetSpec]) -> None:
    """Download datasets to the HF hub cache before submitting SLURM jobs.

    Uses snapshot_download (same as pre_download_models) so datasets land in
    HF_HUB_CACHE and are available when running with HF_HUB_OFFLINE=1.
    """
    cache_dir = os.environ.get("HF_HUB_CACHE")
    logger.info(
        "Pre-downloading %d dataset(s) to %s...",
        len(datasets),
        cache_dir or "default cache",
    )
    for spec in datasets:
        split = spec.get("split")
        label = spec["path"] + (f" [{split}]" if split else "")
        print("Pre-downloading dataset %s..." % label)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(spec["path"], repo_type="dataset", cache_dir=cache_dir)
            logger.info("  %s: OK", label)
        except Exception:
            logger.exception("  %s: download failed", label)
