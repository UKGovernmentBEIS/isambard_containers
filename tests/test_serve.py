"""Tests for vllm-serve CLI and SLURM template integration."""

from __future__ import annotations

from isambard_container_tools.engines.vllm.serve import (
    SBATCH_TEMPLATE,
    _parse_extra_args,
    format_script_args,
)


def test_extra_args_mixed_styles() -> None:
    """CLI passthrough handles --key value, --key=value, and bare --flag."""
    result = _parse_extra_args(
        [
            "--tp",
            "4",
            "--max-model-len=16384",
            "--enable-prefix-caching",
        ]
    )
    assert result["tp"] == 4
    assert result["max_model_len"] == 16384
    assert result["enable_prefix_caching"] is True


def test_script_args_formatting() -> None:
    """Dict → CLI args for user scripts handles all value types."""
    result = format_script_args(
        {
            "model": "foo/bar",
            "verbose": True,
            "quiet": False,
            "skip": None,
            "tags": ["a", "b"],
        }
    )
    assert "--model" in result
    assert "--verbose" in result
    assert "quiet" not in result
    assert "skip" not in result
    assert "--tags a b" in result


def test_template_uses_listen_host_not_hardcoded() -> None:
    """SLURM template uses LISTEN_HOST variable, not hardcoded 0.0.0.0."""
    content = SBATCH_TEMPLATE.read_text()
    assert "LISTEN_HOST" in content, "template missing LISTEN_HOST"
    assert "--host 0.0.0.0" not in content, "template has hardcoded 0.0.0.0"


def test_template_supports_gimlet() -> None:
    """SLURM template has optional gimlet tunnel support."""
    content = SBATCH_TEMPLATE.read_text()
    assert "GIMLET_TOKEN_FILE" in content, "template missing gimlet"
    assert "gimlet-agent" in content, "template missing gimlet-agent launch"


def _stub_images(*names: str) -> list:
    """Minimal stand-ins for sifter LocalImage objects (only `.name` is read)."""
    from types import SimpleNamespace

    return [SimpleNamespace(name=n) for n in names]


def test_latest_family_build_picks_highest_version_excluding_subfamilies(monkeypatch):
    """`vllm` resolves to the highest `vllm-<version>`, ignoring `vllm-lens-*` / `*-head`."""
    from isambard_container_tools.engines.vllm import serve

    monkeypatch.setattr(
        serve.api,
        "list_local_sifs",
        lambda: _stub_images(
            "vllm-0.22.1",
            "vllm-0.23.0",
            "vllm-lens-0.23.0",
            "vllm-lens-head-7-jun-2026",
            "sglang-0.5.6",
        ),
    )
    assert serve._latest_family_build("vllm") == "vllm-0.23.0"
    assert serve._latest_family_build("vllm-lens") == "vllm-lens-0.23.0"


def test_latest_family_build_raises_when_family_absent(monkeypatch):
    """No matching family member -> FileNotFoundError (callers turn it into an exit)."""
    import pytest

    from isambard_container_tools.engines.vllm import serve

    monkeypatch.setattr(
        serve.api, "list_local_sifs", lambda: _stub_images("sglang-0.5.6")
    )
    with pytest.raises(FileNotFoundError):
        serve._latest_family_build("vllm")


def test_find_latest_container_targets_versioned_build(monkeypatch):
    """A pinned version maps to the `<family>-<version>` build; a leading `v` is stripped."""
    from isambard_container_tools.engines.vllm import serve

    monkeypatch.setattr(serve.api, "latest", lambda name: f"/sifs/{name}.sif")
    assert serve.find_latest_container("v0.23.0") == "/sifs/vllm-0.23.0.sif"
    assert (
        serve.find_latest_container("0.23.0", vllm_lens=True)
        == "/sifs/vllm-lens-0.23.0.sif"
    )
