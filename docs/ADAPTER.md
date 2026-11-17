# The shared adapter — principle & training recipe (weights not shipped)

We **do not redistribute trained adapter weights**. This doc gives the principle and a recipe so you
can train your own on your own base model / domain. (License note below: redistribution would be
permitted anyway — the base is Apache-2.0 — but shipping the method, not the artifact, keeps this repo
clean and domain-agnostic.)

## What it is
A tiny **shared, customer-agnostic residual read/write interface** on top of a **frozen** base LM.
- Per injection layer `li`, two matrices `A_K[li]`, `A_V[li]`, each `head_dim × head_dim`.
- For head_dim=256 and 4 injection layers: `4 × 2 × 256 × 256 = 524,288` params (~2 MB) — ~0.06% of a 0.8B base.
- Applied at injection time: `K' = K + s·(K @ A_K)`, `V' = V + s·(V @ A_V)` (s = injection_scale, default 1).
- **Zero-initialized** (`torch.zeros`), so at step 0 injection is the raw fact K/V; training learns the residual.

## Why it's needed
A frozen model only "reads" its own K/V format. Raw fact K/V prepended to attention is glanced at but not
used. The adapter transforms injected K/V into a form the model attends to and acts on.
Measured (W62): **without adapter injection utilization ~17.5%; with adapter ~95%.** It is the bridge from
"memory present" to "memory used".

## Training principle
Only the adapter trains; the base stays frozen (`requires_grad_(False)` on all base params).
1. Encode a fact's value span into per-layer K/V rows (frozen forward) — the memory.
2. Build a query whose answer is that fact's value; inject the fact K/V (through the adapter) into attention.
3. Compute the model's logits over the **answer tokens**; loss = `cross_entropy(logits, answer_ids)`.
4. Backprop **only into `A_K`/`A_V`**; optimizer step. (Adapter math in float32 for stable grads.)

Objective: make the injected memory *drive the model to emit the remembered value*. The adapter learns a
read interface, not new knowledge — knowledge lives in the external memory rows.

## Recipe (reference hyperparameters)
- base: frozen Qwen3.5-0.8B / 4B (Apache-2.0); injection layers e.g. (11,15,19,23), ret_layer/ret_head per model.
- render: **natural sentence** (match the render you'll use at inference — render mismatch costs ~25–35 pts, W72/W109).
- data: fact corpus (e.g. DuIE/CMeIE, or your domain facts), ~200 entities/corpus, value length ≤ 24 tokens.
- optimizer: lr 1e-3, ~3000 steps, zero-init (or warm-start from a template-trained adapter).
- trainable: 524,288 params; everything else frozen.

## License
Base Qwen3.5 weights are **Apache-2.0** (0.8B and 4B checked; no restrictive Qwen monthly-active/commercial
clauses). Apache-2.0 permits derivative works and redistribution with attribution + license + change notices.
So a trained adapter *could* be redistributed under Apache-2.0; we nonetheless ship **principle only**, not
weights, by project decision. Base weights themselves are never bundled — download them under their license.