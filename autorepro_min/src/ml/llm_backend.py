"""
AutoRepro-Min: LLM Backend

Loads an open-source code LLM via Hugging Face `transformers` and exposes a
single `rank()` method used by LLMPrioritizer. Everything torch/transformers
is imported lazily so the default install (no `[llm]` extras) never touches
them.

The prompt asks the model to output a strict JSON array of unit indices,
sorted by "most-likely-irrelevant to least-likely-irrelevant". The parser
tolerates a fair amount of formatting slop (fenced code blocks, prose
before/after the JSON, trailing commas) because small models are less
disciplined than frontier ones.
"""

from __future__ import annotations

import json
import re
import sys as _sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from parser import CodeUnit
from .model_registry import ModelSpec, resolve
from .prioritizers import ErrorContext


# -------- Prompt template ---------------------------------------------------

_SYSTEM_PROMPT = """You are a code-triage assistant helping to minimize a \
Python bug reproduction.

You will be given (1) the error the reproduction produced, and (2) a list \
of candidate code units that could potentially be deleted.

Your job: rank the units from MOST LIKELY IRRELEVANT to the error, to \
LEAST LIKELY IRRELEVANT. Units that are clearly dead code (unused \
helpers, unused imports, prints, unrelated classes) should come first. \
Units that appear in the stack trace or that define symbols referenced \
by traced code should come last.

Respond with ONLY a JSON array of integer unit IDs in your recommended \
removal order, e.g. [3, 0, 5, 1, 2, 4]. No prose, no code fences, no \
explanation."""


_USER_TEMPLATE = """### Error context
{context}

### Candidate units
{units}

### Task
Output a JSON array of unit IDs ordered from most to least likely \
irrelevant to the error above."""


# -------- Unit serialization ------------------------------------------------

def _snippet(text: str, max_chars: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _serialize_units(units: List[CodeUnit]) -> str:
    """Compact JSON-ish list the model consumes."""
    rows: List[str] = []
    for i, u in enumerate(units):
        rows.append(json.dumps({
            "id": i,
            "type": u.node_type,
            "lines": f"{u.start_line}-{u.end_line}",
            "size": u.size,
            "executed": bool(u.execution_count),
            "snippet": _snippet(u.text or ""),
        }, ensure_ascii=False))
    return "[\n  " + ",\n  ".join(rows) + "\n]"


# -------- Response parsing --------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def parse_ranking(raw: str) -> Optional[List[int]]:
    """Extract the first JSON array of ints from a possibly-messy response."""
    if not raw:
        return None

    # Fast path: response IS the JSON array.
    stripped = raw.strip()
    for candidate in [stripped, *_JSON_ARRAY_RE.findall(stripped)]:
        try:
            data = json.loads(candidate.replace(",]", "]"))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        result: List[int] = []
        for x in data:
            if isinstance(x, bool):  # bools are ints in Python — skip
                continue
            if isinstance(x, int):
                result.append(x)
            elif isinstance(x, str) and x.strip().lstrip("-").isdigit():
                result.append(int(x.strip()))
        if result:
            return result
    return None


# -------- Backend -----------------------------------------------------------

@dataclass
class HFBackend:
    """transformers-based LLM backend.

    Loads the model+tokenizer once, then answers `rank()` calls by
    applying the chat template, generating a short response, and parsing
    the ranked JSON list.
    """
    spec: ModelSpec
    model: Any            # AutoModelForCausalLM instance
    tokenizer: Any        # AutoTokenizer instance
    device: str = "cpu"
    max_new_tokens: int = 256
    verbose: bool = False

    def rank(self, context: ErrorContext,
             units: List[CodeUnit]) -> List[int]:
        import torch  # lazy

        prompt_user = _USER_TEMPLATE.format(
            context=context.summarize(),
            units=_serialize_units(units),
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,     # ignored when do_sample=False
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id),
            )

        # Slice off the prompt tokens; only decode the model's completion.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if self.verbose:
            print(f"[HFBackend {self.spec.tier}] raw response: {raw!r}")

        ranking = parse_ranking(raw)
        if ranking is None:
            raise ValueError("model did not return a parseable ranking")
        return ranking


def _select_device(preference: Optional[str] = None) -> str:
    """Pick the best available torch device."""
    import torch  # lazy
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cuda_supports_bf16() -> bool:
    """True on Ampere (compute 8.0) and later; False on Turing / older."""
    import torch  # lazy
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def _load_kwargs(device: str, quantize_requested: bool, spec, verbose: bool):
    """Return kwargs for AutoModelForCausalLM.from_pretrained().

    Selection policy:
      * quantize_requested is True (the CLI default) AND the model's own
        spec.quantize_by_default is True AND we're on CUDA with bnb
        available  ->  4-bit NF4 (compute bf16 on Ampere+, fp16 on Turing)
      * everything else  ->  full precision for the device
                             (fp32 on CPU, fp16 on MPS/CUDA)

    So `tiny`/`small` load full-precision by default (they fit anywhere
    at fp16 and quantization loss hurts them more), while
    `medium`/`large`/`alt` load 4-bit by default. --no-quantize forces
    full precision on all tiers.
    """
    import torch  # lazy

    if device == "cpu":
        return {"torch_dtype": torch.float32}, "fp32/cpu"

    if device == "mps":
        # bitsandbytes doesn't support MPS. fp16 is the reasonable default.
        return {"torch_dtype": torch.float16}, "fp16/mps"

    # device == "cuda"
    should_quantize = quantize_requested and spec.quantize_by_default
    if should_quantize:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401 -- import check
        except ImportError:
            if verbose:
                print("[llm_backend] bitsandbytes not installed; "
                      "loading in fp16 instead. Install with "
                      "`pip install bitsandbytes` for 4-bit.")
        else:
            compute_dtype = (torch.bfloat16 if _cuda_supports_bf16()
                             else torch.float16)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            # device_map="auto" lets accelerate/bitsandbytes place the
            # quantized shards; passing .to("cuda") after 4-bit load is
            # unsupported.
            return (
                {"quantization_config": bnb, "device_map": "auto"},
                f"4-bit NF4 (compute {compute_dtype})",
            )

    # Full-precision CUDA path (either --no-quantize, small model, or
    # bitsandbytes unavailable).
    dtype = torch.bfloat16 if _cuda_supports_bf16() else torch.float16
    label = f"{dtype}/cuda"
    if quantize_requested and not spec.quantize_by_default:
        label += " (small model — quantization skipped by policy)"
    return {"torch_dtype": dtype}, label


def build_llm_backend(model: Optional[str] = None,
                      device: Optional[str] = None,
                      quantize: bool = True,
                      verbose: bool = False) -> Optional[HFBackend]:
    """Try to build an HFBackend for the given tier.

    Args:
        model: tier name from model_registry (default: "small").
        device: force a torch device; auto-detected if None.
        quantize: on CUDA, load in 4-bit NF4 via bitsandbytes when
            available. Ignored on MPS/CPU (bitsandbytes doesn't support
            them). Default True — 4-bit code-model accuracy loss is
            small enough that there's no reason to pay full precision.
        verbose: print progress + selection reasoning.

    Returns None (with a warning) if torch/transformers isn't installed
    or the model can't be loaded — callers should treat None as
    "LLM path unavailable" and fall back to the heuristic.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        if verbose:
            print(f"[build_llm_backend] ML extras not installed ({exc}); "
                  "install with `pip install .[llm]`")
        return None

    try:
        spec = resolve(model)
    except KeyError as exc:
        if verbose:
            print(f"[build_llm_backend] {exc}")
        return None

    dev = _select_device(device)
    load_kwargs, dtype_label = _load_kwargs(dev, quantize, spec, verbose)
    if verbose:
        print(f"[build_llm_backend] loading {spec.hf_id} on {dev} "
              f"({dtype_label})...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        model_obj = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, **load_kwargs)
        # 4-bit loads use device_map and are already placed. For everything
        # else, move to the target device.
        if "device_map" not in load_kwargs:
            model_obj.to(dev)
        model_obj.eval()
    except Exception as exc:
        if verbose:
            print(f"[build_llm_backend] failed to load {spec.hf_id}: {exc}")
        return None

    return HFBackend(spec=spec, model=model_obj, tokenizer=tokenizer,
                     device=dev, verbose=verbose)
