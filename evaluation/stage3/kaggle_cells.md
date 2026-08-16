# Stage 3 generation notebook

The GPU half of the experiment, and deliberately the dumb half. Every
decision that could bias the comparison — which files each arm sees, how
the budget is spent, what the prompt says — is made by
`build_contexts.py` on CPU and committed to the repository. This
notebook loads that file and runs one model over it.

Pushed with the kaggle MCP server:

    kaggle_push_notebook(slug="automre-stage3", cells_file=".../kaggle_cells.md")

Results land in `/kaggle/working/generations.jsonl`, one line per
(instance, arm, sample), written as they are produced so a session that
runs out of time still returns everything up to that point.

## Cell 0 — configuration

```python
import os, json, time, subprocess, sys

REPO   = "https://github.com/i-shantt/autoMRE"
BRANCH = "stage3-reduced-context"
MODEL  = "Qwen/Qwen2.5-Coder-7B-Instruct"   # Apache-2.0, open weights

K_SAMPLES      = 5        # samples per (instance, arm)
TEMPERATURE    = 0.6      # Qwen's own recommendation for this model
TOP_P          = 0.95
MAX_NEW_TOKENS = 1024

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
print(f"{len(contexts)} contexts")
for c in contexts:
    print(f"  {c['instance_id']:<24} {c['arm']:<13} "
          f"{c['prompt_tokens']:>6} prompt tokens  "
          f"{len(c['included_files']):>3} files  "
          f"gold in context: {bool(c['gold_files_in_context'])}")
```

## Cell 2 — the model

```python
import torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers", "accelerate"], check=False)
from transformers import AutoModelForCausalLM, AutoTokenizer

print("GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto",
    attn_implementation="sdpa")
model.eval()
print("loaded", MODEL)

# Samples for one context share a single prefill. At ~17k prompt tokens
# that is most of the cost, so five samples take barely longer than one;
# generating them one at a time would pay the prefill five times over.
def generate(prompt, n):
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
```

## Cell 3 — generate

```python
# On out-of-memory the batch halves and the work is retried rather than
# lost — a T4 that got a smaller share of memory should slow the run
# down, not end it.

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
    texts, batch = [], len(todo)
    while len(texts) < len(todo):
        try:
            texts += generate(ctx["prompt"], min(batch, len(todo) - len(texts)))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            print(f"  OOM -> batch {batch}")
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
prompt = ("### synthetic.py\n```python\n"
          + "".join(f"def f_{i}(x):\n    return x + {i}\n\n"
                    for i in range(2200))
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
