# Stage 3 generation notebook

The GPU half of the experiment, and deliberately the dumb half. Every
decision that could bias the comparison — which files each arm sees, how
the budget is spent, what the prompt says — is made by
`build_contexts.py` on CPU and committed to the repository. This
notebook loads that file and runs one model over it.

Two invocations, and the cell lists are not interchangeable:

    # the capacity check — run this first, it is two minutes
    python3 evaluation/stage3/push_kernel.py \
        --slug automre-stage-3-capacity-check \
        --include "Cell 0,Cell 2,Cell 5"

    # the run itself. Name the cells; do NOT push without --include,
    # because Cell 5 is last in document order and would run the
    # capacity check *after* the generations, where an out-of-memory
    # would mark the whole kernel ERROR on top of work that succeeded.
    python3 evaluation/stage3/push_kernel.py --slug automre-stage3 \
        --include "Cell 0,Cell 1,Cell 2,Cell 3,Cell 4"

Both take `--kaggle <path to the kaggle CLI>`. Results land in
`/kaggle/working/generations.jsonl`, one line per (instance, arm,
sample), written as they are produced so a session that runs out of
time still returns everything up to that point, and fetched with
`kaggle kernels output`.

## Cell 0 — configuration

```python
import os, json, time, subprocess, sys

REPO   = "https://github.com/i-shantt/autoMRE"
BRANCH = "stage3-reduced-context"
MODEL  = "Qwen/Qwen2.5-Coder-7B-Instruct"   # Apache-2.0, open weights

K_SAMPLES      = 5        # samples per (instance, arm)
TEMPERATURE    = 0.6      # Qwen's own recommendation for this model
TOP_P          = 0.95
# Measured against the answers, not guessed. Re-expressed as
# SEARCH/REPLACE edits, the ground-truth patches run 105 to 719 tokens,
# and pylint's 719 is 70% of a 1024 ceiling — close enough that a model
# adding a sentence of preamble would be cut off mid-block and scored
# as having produced nothing parseable. Generation stops at EOS anyway,
# so the headroom is free.
MAX_NEW_TOKENS = 1536

OUT = "/kaggle/working/generations.jsonl"
print("config ok")
```

## Cell 1 — the repository and the contexts

```python
if not os.path.exists("/kaggle/working/autoMRE"):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO, "/kaggle/working/autoMRE"], check=True)

CTX = "/kaggle/working/autoMRE/evaluation/stage3/contexts.jsonl"
contexts = [json.loads(l) for l in open(CTX) if l.strip()]

# An arm whose context came out empty is not a hard question, it is an
# arm that could not be built: the gold file is larger than the budget,
# so `pack` — which only takes whole files — took nothing. Generating
# against it would spend GPU time producing answers to a prompt with no
# repository in it, and score them as a model that failed.
skipped = [c for c in contexts if c.get("context_empty")]
contexts = [c for c in contexts if not c.get("context_empty")]
for c in skipped:
    print(f"skipping {c['instance_id']} {c['arm']}: no file fits whole "
          f"(oversize gold: {', '.join(c['gold_files_oversize']) or 'none'})")

# An instance whose controls failed cannot be scored, so generating for
# it spends GPU time on output nothing will ever read. seaborn-3187 is
# the one here: pandas 3.0 against 2022-era seaborn breaks 249
# previously-passing tests, so even the ground-truth patch reads as a
# regression. Its controls are committed beside this file as the
# evidence for the exclusion, rather than the instance quietly going
# missing.
scoreable = set(json.load(open(
    "/kaggle/working/autoMRE/evaluation/stage3/controls.json"))["scoreable"])
dropped = sorted({c["instance_id"] for c in contexts
                  if c["instance_id"] not in scoreable})
contexts = [c for c in contexts if c["instance_id"] in scoreable]
for iid in dropped:
    print(f"skipping {iid}: controls failed, nothing here could score it")

print(f"{len(contexts)} contexts")
for c in contexts:
    print(f"  {c['instance_id']:<24} {c['arm']:<13} "
          f"{c['prompt_tokens']:>6} prompt tokens  "
          f"{len(c['included_files']):>3} files  "
          f"gold in context: {bool(c['gold_files_in_context'])}")
```

## Cell 2 — the model

```python
# Which GPU this session got, asked *before* importing torch — because
# the answer decides which torch to import.
#
# Kaggle hands out whichever accelerator is free and `machine_shape` is
# advisory: three consecutive pushes asking for a T4 were all given a
# Tesla P100, and so was a push naming a deliberately invalid shape. So
# the P100 is not an unlucky roll to retry past, it is the machine.
#
# A P100 is compute capability sm_60. Kaggle's preinstalled PyTorch is
# built for sm_70 and up, so on that card the model downloads, loads,
# prints "loaded", and every generation then dies with
# cudaErrorNoKernelImageForDevice — ten minutes in, with nothing saved.
# PyTorch shipped sm_60 through the 2.5 series, so that is what a P100
# session installs.
cap = subprocess.run(
    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.split()
print("compute capability:", cap)

if cap and float(cap[0]) < 7.0:
    print(f"sm_{cap[0].replace('.', '')} — installing a PyTorch built "
          f"for it (a few minutes)", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"],
                   check=True)
    # Pinned against that torch rather than upgraded past it: the newest
    # transformers expects a newer torch than the one just installed.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.49.0", "accelerate==1.4.0"], check=True)
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "transformers", "accelerate"], check=False)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# Belt and braces: if the install above did not produce a torch that can
# actually target this card, say so now rather than after the download.
supported = {int(a.removeprefix("sm_")) for a in torch.cuda.get_arch_list()
             if a.startswith("sm_")}
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    if major * 10 + minor not in supported:
        raise SystemExit(
            f"{torch.cuda.get_device_name(i)} is sm_{major}{minor}; this "
            f"PyTorch supports {sorted(supported)}. Re-run to be given a "
            f"different GPU — nothing here will work on this one.")

tok = AutoTokenizer.from_pretrained(MODEL)

# `device_map="auto"` fills one GPU before touching the next, so on two
# T4s it put 11.5 GiB of weights on GPU 0 and left GPU 1 half empty —
# then the KV cache for five 18k-token sequences had 1.25 GiB to live
# in and asked for 1.31. The weights are capped per device instead, so
# they spread and leave the same headroom on both cards.
#
# 1.66 GiB was also reserved-but-unallocated, i.e. fragmentation, which
# is what expandable segments are for.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Force scaled-dot-product attention onto a backend whose memory is
# linear in sequence length. Left to choose, it fell back to the *math*
# backend and asked for 36.60 GiB on a 14.56 GiB card — which is
# exactly 28 heads x 18702^2 x 4 bytes, the full attention matrix,
# materialised. That is why halving the batch never helped: at batch 1
# the request was unchanged, because the cost is quadratic in the
# prompt, not linear in the batch.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

n_gpu = torch.cuda.device_count()
per_gpu = torch.cuda.get_device_properties(0).total_memory / 2**30
weight_cap = f"{max(4.0, per_gpu * 0.55):.0f}GiB"
print(f"{n_gpu} x {per_gpu:.1f} GiB; capping weights at {weight_cap}/GPU")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto",
    max_memory={i: weight_cap for i in range(n_gpu)},
    attn_implementation="sdpa")
model.eval()
print("loaded", MODEL, "->", set(model.hf_device_map.values()))

# Samples for one context share a single prefill. At ~17k prompt tokens
# that is most of the cost, so five samples take barely longer than one;
# generating them one at a time would pay the prefill five times over.
def _generate_once(prompt, n):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=n,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return [tok.decode(seq[enc["input_ids"].shape[1]:],
                       skip_special_tokens=True) for seq in out]


# How many samples fit at once is not knowable in advance: the prefill
# activation scales with the batch, so it depends on the prompt, and the
# prompts here run from 16k to 17.4k tokens. Halving on OOM finds the
# answer per call rather than making the whole run pay for the worst
# context.
#
# This lives here rather than in the generation loop so the capacity
# check exercises the same code the run does — a check that measures an
# easier thing than the run is not a check.
def generate(prompt, n):
    texts, batch = [], n
    while len(texts) < n:
        try:
            texts += _generate_once(prompt, min(batch, n - len(texts)))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            print(f"  OOM -> batch {batch}", flush=True)
    return texts
```

## Cell 3 — generate

```python
# Results are appended as they are produced, and a restart picks up
# from what is already on disk — a session that runs out of time
# returns everything up to that point rather than nothing. Batch
# shrinking on out-of-memory lives in `generate` itself, so the
# capacity check and this loop behave identically.

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        row = json.loads(line)
        done.add((row["instance_id"], row["arm"], row["sample"]))
    print(f"resuming: {len(done)} generations already on disk")

fh = open(OUT, "a")
start = time.time()
for ctx in contexts:
    todo = [s for s in range(K_SAMPLES)
            if (ctx["instance_id"], ctx["arm"], s) not in done]
    if not todo:
        continue
    t0 = time.time()
    texts = generate(ctx["prompt"], len(todo))
    for sample, text in zip(todo, texts):
        fh.write(json.dumps({
            "instance_id": ctx["instance_id"], "arm": ctx["arm"],
            "sample": sample, "output": text,
            "model": MODEL, "temperature": TEMPERATURE,
            "prompt_tokens": ctx["prompt_tokens"],
        }) + "\n")
    fh.flush()
    print(f"{ctx['instance_id']:<24} {ctx['arm']:<13} "
          f"{len(todo)} samples in {time.time()-t0:.0f}s "
          f"(total {(time.time()-start)/60:.1f} min)", flush=True)
fh.close()
print("generation complete")
```

## Cell 4 — what came back

```python
rows = [json.loads(l) for l in open(OUT)]
print(f"{len(rows)} generations")
have_block = sum("<<<<<<< SEARCH" in r["output"] for r in rows)
print(f"{have_block} contain a SEARCH/REPLACE block "
      f"({have_block / max(len(rows), 1) * 100:.0f}%)")
for arm in sorted({r["arm"] for r in rows}):
    got = [r for r in rows if r["arm"] == arm]
    hit = sum("<<<<<<< SEARCH" in r["output"] for r in got)
    print(f"  {arm:<13} {len(got):>3} generations, {hit:>3} parseable")
print("\n--- one sample ---\n")
print(rows[0]["output"][:1500])
```

## Cell 5 — capacity check

Run this alone (`include="Cell 0,Cell 2,Cell 5"`) before queuing the
real thing. It answers the three questions that decide whether the run
finishes: how many GPUs this session got, whether five samples share one
16k-token prefill without running out of memory, and how long a cell
takes — which times the whole run. A synthetic prompt of the right size
does that without needing `contexts.jsonl` to exist yet.

```python
# Sized against the tokenizer rather than guessed: 2,200 functions came
# to 39,602 tokens, which is past the model's 32,768-token window, so
# the check was testing something the run will never do. Grow until the
# real prompt size is reached and stop there.
TARGET = 17_500        # the largest prompt in contexts.jsonl, rounded up

body, n_tok, i = "", 0, 0
while n_tok < TARGET:
    body += f"def f_{i}(x):\n    return x + {i}\n\n"
    i += 1
    if i % 100 == 0:
        n_tok = len(tok(body)["input_ids"])
prompt = ("### synthetic.py\n```python\n" + body
          + "```\nFix the bug and reply with a SEARCH/REPLACE edit.")
n_tok = len(tok(prompt)["input_ids"])
print(f"synthetic prompt: {n_tok} tokens")

t0 = time.time()
texts = generate(prompt, 5)
dt = time.time() - t0
print(f"5 samples in {dt:.0f}s -> {21 * dt / 60:.0f} min for 21 contexts")
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i} peak {torch.cuda.max_memory_allocated(i)/2**30:.1f} GiB")
print(repr(texts[0][:200]))
```
