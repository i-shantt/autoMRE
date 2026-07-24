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
    ram_4bit_gb: float           # loaded footprint w/ NF4 quantization (CUDA)
    ram_fp16_gb: float           # loaded footprint at fp16 (MPS / non-quant)
    family: str                  # "qwen2.5-coder" or "codegemma"
    quantize_by_default: bool    # 4-bit NF4 on CUDA by default?
    description: str


# Sizes below are ROUGH — actual footprint depends on the tokenizer, KV
# cache, and framework overhead. The 4-bit column applies only on CUDA
# with bitsandbytes installed; MPS (Mac) and CPU paths use the fp16/fp32
# column since bitsandbytes doesn't support them.
#
# Quantization policy: the small models (tiny/small) already fit
# comfortably in 4 GB at fp16, so we run them full-precision by default —
# quantization accuracy loss hurts them more than it helps memory-wise.
# The bigger tiers (medium/large/alt) default to 4-bit on CUDA so a 6 GB
# VRAM card fits every option. Users can override either direction with
# --no-quantize on the CLI.
MODEL_TIERS: Dict[str, ModelSpec] = {
    "tiny": ModelSpec(
        tier="tiny",
        hf_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        params_b=0.5,
        ram_4bit_gb=0.4,
        ram_fp16_gb=1.2,
        family="qwen2.5-coder",
        quantize_by_default=False,
        description="Fastest option. Runs full-precision on any laptop.",
    ),
    "small": ModelSpec(
        tier="small",
        hf_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        params_b=1.5,
        ram_4bit_gb=1.1,
        ram_fp16_gb=3.2,
        family="qwen2.5-coder",
        quantize_by_default=False,
        description=(
            "Recommended balance. Runs full-precision; fits in 4 GB "
            "of VRAM."),
    ),
    "medium": ModelSpec(
        tier="medium",
        hf_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        params_b=3.0,
        ram_4bit_gb=2.1,
        ram_fp16_gb=6.5,
        family="qwen2.5-coder",
        quantize_by_default=True,
        description=(
            "Consumer GPU sweet spot. 4-bit by default so it fits in "
            "6 GB VRAM."),
    ),
    "large": ModelSpec(
        tier="large",
        hf_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        params_b=7.0,
        ram_4bit_gb=4.5,
        ram_fp16_gb=14.0,
        family="qwen2.5-coder",
        quantize_by_default=True,
        description=(
            "Best quality in the menu. 4-bit by default (fits in 6 GB "
            "VRAM tight, 8 GB comfortably); full-precision needs 16 GB+."),
    ),
    "alt": ModelSpec(
        tier="alt",
        hf_id="google/codegemma-2b-it",
        params_b=2.0,
        ram_4bit_gb=1.6,
        ram_fp16_gb=4.5,
        family="codegemma",
        quantize_by_default=True,
        description=(
            "Cross-family control (CodeGemma-2B-it) for the comparison "
            "matrix. 4-bit by default."),
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
