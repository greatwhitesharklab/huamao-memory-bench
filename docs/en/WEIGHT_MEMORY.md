# Per-customer weight memory (brain-style) — a controlled characterization

**English** | [中文](../zh/WEIGHT_MEMORY.md)

Companion to [`CLAIMS.md`](CLAIMS.md) / [`EXPERIMENTS.md`](EXPERIMENTS.md). Evidence:
[`results/weightmem/`](../../results/weightmem/). **Status: mechanism-level, synthetic customers only.**

> **This is a different memory paradigm from the main B-KV line.** The main line keeps customer facts in
> **external, auditable, deletable, rollback-able K/V rows** and trains only a *shared* adapter. This line
> explores the opposite, brain-style commitment: **the customer's knowledge lives in per-customer weights**
> (a LoRA), evolves with use, and is "audited" by asking the model. It was an explicit, owner-approved
> relaxation of the "shared adapter / external auditable memory" constraint. **Accepted costs: not cleanly
> deletable, not rollback-able, will forget, self-report ≠ stored state.** Do not conflate the two lines.

## 1. Setup
- Frozen shared base **Qwen3.5-9B**; per-customer **LoRA on `q_proj`/`v_proj`** (rank 16, ~2.2M trainable
  params). The LoRA *is* the customer's memory.
- **Recall = normal generation. No external K/V injection at recall time.** The preference must come from weights.
- Training objective = the validated "use, don't just read" recipe: grouped counterfactuals (preferred label
  varies per pair; read order varies) + correct-action margin loss. This avoids positional shortcuts.
- Task: a customer prefers one of two arbitrary labels per pair; a question maps the labels to two actions
  (上午/下午, or 左边/右边); the model must pick the action holding the preferred label.

## 2. What works (mechanism evidence)

| Property | Result | Reading |
|---|---|---|
| Weight memory vs frozen base | choice **8/8** vs 4/8; mapping-flip **4/4** vs 0/4; order-robust read **4/4** vs 2/4 | knowledge is in weights and **used**, not a positional shortcut (`validation_v2.json`) |
| Capacity | **150 facts at once → choice/flip/read all 1.0** | rank-16 holds ≥150 facts; capacity not the bottleneck here (`capacity_150.json`) |
| Generalization / migration | orig / rephrase / **new action words (左边/右边+展位)** all **6/6 flips** | preference is abstract enough to apply to unseen action vocabulary (`migration.json`) |
| Similar-fact discrimination | same labels, different preference per context (场次 vs 展位): **1.0** | context-dependent memories stay distinct (`probe_context_disc.json`) |
| Memory revision | switch preference A→B: new **0.92**, read 1.0; old intrusion **8%** | updatable; mild proactive interference (`probe_revision.json`) |
| Forgetting + replay | see §3 | replay eliminates forgetting at both scales |

## 3. Forgetting (continual, no-replay vs replay; retention of the oldest batch)

| | 60 facts / 4 rounds | 90 facts / 6 rounds |
|---|---|---|
| **no replay** | oldest **0.87** (old-batch mean 0.96) | oldest **0.67** (mean 0.89) |
| **replay** (rehearse a sample of old facts each round) | **1.00** | **1.00** |

No-replay forgetting **worsens with scale** (0.87→0.67); **replay fully suppresses it** at both scales.
Acquisition of each new batch stays ~1.0 throughout. Evidence: `forgetting_{none,replay}_{60,90}.json`.

## 4. Costs / negative results (reported, not buried)

- **Few-shot USE is slow.** After 1/2/4 exposures of a fact: read reaches 1.0 by 4, but **choice stays at
  chance (0.5), flip 0**. Readable memory forms fast; *usable* memory needs repetition. (`probe_one_shot.json`)
- **Audit-by-asking fails for enumeration.** Asked to "list all the customer's preferences", open-ended
  **recall = 0.0**: the model *denies having the data* ("you haven't provided it / I can't access your records"),
  even though **every cued choice/read is correct**. Knowledge is **usable-but-not-reportable**; the model
  confabulates a reason rather than introspecting. This sharply quantifies the accepted "self-report ≠ state"
  cost. (`audit.json`)
- **By design:** not cleanly deletable, not rollback-able (knowledge is in weights).

## 5. Scope and limits
- **Synthetic customers, ≤150 arbitrary-label facts, rank 16, one base size (9B).** Mechanism evidence only;
  **not** a production-capability claim. Real-corpus validation is pending (project discipline).
- The "preference" is an arbitrary label association, not rich real-world preference semantics.
- Forgetting conclusions are at this scale; hundreds/thousands of facts or far more rounds may differ.

## 6. Related work / positioning — Piccolo
[AgentScope's Piccolo](https://github.com/agentscope-ai/Piccolo) (Apache-2.0) is a family of **4B/9B/27B
decision models on the same Qwen3.5 base**: given state/question/candidates it reads candidate-token logits
with bidirectional full-attention prompt attention and returns a structured choice + probabilities, **generating
no answer text** (native `/v1/systemone`, evaluated on JevBench). It validates the "small model + logit-read
decision, no text" interface this project also uses. **Difference / niche:** Piccolo is a *general decision
engine* trained on millions of in-context decision examples and does **not** do per-customer memory; this line
contributes the **memory** side (per-customer knowledge that drives decisions). They are complementary
(memory feeds a decision head), not competing on general decision-making.

## 7. Reproduce
Recipe + per-arm curves/probes in [`results/weightmem/`](../../results/weightmem/) (`suite_config.json` has
hyperparameters). Weights are **not shipped** (bring-your-own Qwen3.5-9B); LoRA training is standard
AdamW on the use-objective above.
