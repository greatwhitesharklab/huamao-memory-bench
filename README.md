# huamao-bkv — native writable K/V memory for a frozen small LM

**English** | [中文](README.zh-CN.md)

**花猫 (huamao, "flower cat")**: build writable, overwritable, fuzzily-retrievable memory **outside** a
frozen small language model — without growing its parameters or context window. Recall injects the
remembered facts as **K/V rows** into the model's attention; the model's own decode converges to an answer.

> **Status: mechanism library + reproducible benchmark. Capability claims are pending.**
> Read [`CLAIMS.md`](CLAIMS.md) before citing any number — negative results are listed there too.

## What's in the box
- `huamao_bkv/` — the B-KV engine (`BKVMemory`): frozen base + full-attn-layer K/V cache + a small
  **shared** residual adapter + `ent_mean` fuzzy retrieval + K/V-injection generation.
  - **Conflict overwrite**: structured primary key `(entity, attribute)`, latest wins, with a
    `superseded` audit trail (inspectable / rollback-able). No text parsing in the mechanism.
  - **Natural-sentence encoding**: char-offset→token value span (language-agnostic, no hardcoded template).
- `benchmarks/mab/` — reproducible **MemoryAgentBench** (Conflict_Resolution, single-hop-6k) mechanism
  benchmark, runs on **0.8B/4B CPU**.
- `tests/` — offline unit tests (no model weights / GPU needed).

## Quickstart (0.8B CPU)
```bash
pip install -e .                 # torch + transformers
pip install pandas pyarrow huggingface_hub   # benchmark extras

# 1) base weights are NOT bundled — download Qwen3.5-0.8B yourself (ModelScope or HF), e.g.
#    modelscope download Qwen/Qwen3.5-0.8B --local_dir ./models/Qwen3.5-0.8B
# 2) benchmark data (MIT, downloaded not committed):
python scripts/download_mab.py
# 3) run (point --adapter at a trained shared adapter; see "Adapter" below):
python benchmarks/mab/run.py --model ./models/Qwen3.5-0.8B --adapter /path/adapter.pt \
    --data data/mab/sh6k.json --ret-layer 11 --ret-head 2 --n 40
```
Ablations: `--no-overwrite` (append-only), `--template` (legacy Chinese-template encoding).

## Adapter & weights (license)
- **Base Qwen weights are not redistributed** — download them under their own license (Qwen3.5 is Apache-2.0).
- The **shared adapter** (~524k params) is the read interface that makes the model *use* injected K/V
  (without it, utilization drops ~95%→~17.5%). **We ship the principle + training recipe, not trained
  weights** — see [`docs/ADAPTER.md`](docs/ADAPTER.md) to train your own on your base/domain. The mechanism
  runs with a zero adapter (`--adapter` omitted) but injection utilization drops sharply.
- License checked: base Qwen3.5 (0.8B/4B) is **Apache-2.0** (no restrictive Qwen clauses), which permits
  derivative-adapter redistribution; we ship principle-only by decision, not because of a license blocker.

## How it works (one paragraph)
Write a fact → encode its value span into per-layer K/V rows (frozen forward) → store rows + a retrieval
key. At query time, `ent_mean` scores stored keys against the query's entity positions, returns top-k;
their K/V (plus the shared adapter residual) is prepended to attention during prefill/decode; the model
generates. Conflicting updates overwrite the same primary key (audit kept). A recall **threshold** gates
injection so irrelevant facts don't pollute unrelated answers.

## Related work (positioning)
- **RETRO** (DeepMind): retrieval-augmented *pretraining* with chunked cross-attention. B-KV is post-hoc:
  writable K/V injected into an already-frozen model, no pretraining.
- **Engram**: conditional memory via n-gram hash lookup (trained). B-KV uses fuzzy K/V retrieval + injection.
- **NVIDIA arXiv:2611.05306**: model self-compresses text-state for re-injection (our "C arm" candidate).
- **Benchmarks**: LongMemEval, Mem-PAL, and **MemoryAgentBench** (arXiv:2507.05257, MIT) — we use MAB
  Conflict_Resolution to probe update/overwrite + counterfactual injection.

## Honest boundary
Memory helps **knowledge/recall-bound** tasks (small model + memory can approach big-model readout).
It does **not** buy multi-hop reasoning capacity, and on reasoning-bound tasks native injection may not be
semantically used at all. See [`CLAIMS.md`](CLAIMS.md).

## License
Apache-2.0 (this repo). Benchmark data: MIT (MemoryAgentBench). Base model weights: their own license,
not bundled.