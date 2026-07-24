"""
AutoRepro-Min: Model Registry

Declares the open-source models supported by the LLM prioritizer, keyed
by user-facing tier names (tiny / small / medium / large / alt).

Why these choices:

* Qwen2.5-Coder-{0.5B, 1.5B, 3B, 7B}-Instruct — currently the strongest
  open code-instruct family at these scales, with a unified chat template
  so the same prompt code works across tiers. This makes the tier-to-tier
  comparison in the README a genuine apples-to-apples ablation.
* CodeGemma-2B-it — a cross-family control point at roughly the same
  parameter count as `small`/`medium`. If Qwen dominates, we want to
  know whether that's the family or just the size.

All entries live locally in the Hugging Face cache once downloaded; no
API keys, no billing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    tier: str                    # short user-facing name
    hf_id: str                   # Hugging Face repository id
    params_b: float              # parameter count in billions
    approx_ram_gb: float         # rough loaded footprint (bf16)
    family: str                  # "qwen2.5-coder" or "codegemma"
    description: str


MODEL_TIERS: Dict[str, ModelSpec] = {
    "tiny": ModelSpec(
        tier="tiny",
        hf_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        params_b=0.5,
        approx_ram_gb=1.5,
        family="qwen2.5-coder",
        description="Fastest option. Runs on any laptop CPU.",
    ),
    "small": ModelSpec(
        tier="small",
        hf_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        params_b=1.5,
        approx_ram_gb=3.5,
        family="qwen2.5-coder",
        description="Recommended balance. CPU-runnable, real code reasoning.",
    ),
    "medium": ModelSpec(
        tier="medium",
        hf_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        params_b=3.0,
        approx_ram_gb=7.0,
        family="qwen2.5-coder",
        description="Consumer GPU sweet spot. Notably stronger than small.",
    ),
    "large": ModelSpec(
        tier="large",
        hf_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        params_b=7.0,
        approx_ram_gb=15.0,
        family="qwen2.5-coder",
        description="Best quality in the menu. Needs 16GB+ VRAM for speed.",
    ),
    "alt": ModelSpec(
        tier="alt",
        hf_id="google/codegemma-2b-it",
        params_b=2.0,
        approx_ram_gb=5.0,
        family="codegemma",
        description=(
            "Cross-family control (CodeGemma-2B-it) for the comparison "
            "matrix."),
    ),
}


DEFAULT_TIER: str = "small"


def resolve(name: Optional[str]) -> ModelSpec:
    """Look up a tier by name; raise a helpful error if unknown."""
    key = (name or DEFAULT_TIER).lower()
    if key not in MODEL_TIERS:
        available = ", ".join(sorted(MODEL_TIERS.keys()))
        raise KeyError(
            f"unknown model tier {name!r}; available: {available}")
    return MODEL_TIERS[key]


def list_tiers() -> List[ModelSpec]:
    """Ordered list of tiers, small-to-large then alt."""
    order = ["tiny", "small", "medium", "large", "alt"]
    return [MODEL_TIERS[t] for t in order if t in MODEL_TIERS]
