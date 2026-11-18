# Experiments — natural-era results, arm comparisons, honest boundaries

**English** | [中文](../zh/EXPERIMENTS.md)

Companion to [`CLAIMS.md`](CLAIMS.md). Raw 4B MAB result JSONs in [`results/mab/`](../../results/mab/).

> **Era note (important):** earlier template-era numbers (Chinese-frame encoding) are **superseded** —
> W109 showed the template was a bottleneck. Everything below is **natural-sentence encoding** (the current
> default) unless explicitly marked *template-era preliminary*. MAB is a real benchmark (MIT); results are
> mechanism-level. Production capability remains pending real corpus.

## The thesis
**Memory benefit = task type × base capacity.** Memory amplifies what the base can already do; it does not
create reasoning capacity, and cannot rescue what the base refuses/cannot answer.

## Arms
N = no memory · P = full history as text · H = **B-KV native K/V injection** · H2 = discrete top-1 readout ·
Hwrong = poisoned injection (control) · R = text-recall · RAG = BM25 session retrieval (text) · SUM = extractive summary.

## 1. What the mechanism does (MAB, natural era, reproducible)
MemoryAgentBench Conflict_Resolution sh_6k (single-hop, 70% conflict groups). `benchmarks/mab/run.py`.

### Conflict resolution = write-time overwrite (W108)
Structured primary key `(entity, attribute)`, latest wins, `superseded` audit. Conflict top-1:
| base | append-only | overwrite |
|---|---|---|
| 4B | 0.35 | **0.55** |
| 0.8B | 0.45 | **0.825** |

### De-templating = natural-sentence encoding (W109)
Char-offset→token value span (455/455 pure), no hardcoded frame. 4B:
| subset | template | natural |
|---|---|---|
| clean (n=25) | 0.84 | **0.92** |
| conflict (n=40) | 0.55 | **0.80** |

0.8B natural: clean **0.92**, conflict **0.825**. Clean recall 0.92 ≈ 27B oracle-readout anchor 0.925 (W91) —
**level-skipping on knowledge/recall-bound tasks comes from the memory mechanism, not the base** (0.8B≈4B on
easy clean recall; 4B's edge is anti-dilution).

### Base-capability regression needs threshold gating (W110)
10 unrelated questions: N=1.0; forced irrelevant injection H=0.8 (pollution); unrelated recall 0.177–0.263 vs
related ~0.29–0.34 → threshold 0.27 gates all unrelated → H=1.0.

## 2. Arm comparison on recall-bound fact recall (MAB clean, natural era, 0.8B, n=15)
| arm | acc |
|---|---|
| N no memory | 0.20 |
| **H native injection** | **0.933** |
| RAG retrieved-facts-as-text | **1.00** |
| P all-facts-as-text | **1.00** |

**Honest reading:** on a *small, clean, single-hop* fact set, giving the fact as **text is as good or slightly
better** than native injection. Native injection is competitive (0.933) but does **not** beat text here.
Native's advantages are **selectivity** (long/noisy/over-context history), **writability/overwrite**, and
**not consuming prompt tokens** — these need the right scenario to show, not a small clean fact set.

## 3. Reasoning-bound preference recommendation (Mem-PAL) — diagnosis + a brain-inspired fix
Mem-PAL = recall a user's long-term preferences (**accept/reject attitudes, often with negation**) from ~30
sessions, then recommend for a vague query.

**Baseline (0.8B, natural era, 8/16 queries):** N 0.335 / P 0.455 / **H 0.305** / RAG 0.425 / SUM 0.334.
Native H trails RAG/P.

**Diagnosis (controlled micro-tests, W113):**
- Recall is **not** the bottleneck — semantic recall retrieves relevant prefs, not rejected ones.
- The bottleneck: **K/V injection cannot do open-ended synthesis.** With *identical* pref content, injection
  yields garbage (0.8B) / fixation-echo (4B), while the same content as **text** synthesizes coherently.
  It is the injection **form**, not capacity (4B does not fix it).
- **Architectural boundary: B-KV injection = fact-memory engine (write / recall / overwrite / constrained
  readout), NOT an open-synthesis engine.**

**Brain-inspired fix — separate memory-readout from reasoning (two-stage):**
store consolidated facts → **per-fact K/V-injection readout** (injection's strength) → read-out facts as
**text** → reason/recommend (text's strength). Memory recall ≠ reasoning, as in the brain.

**Single-question proof-of-concept (4B, n=1, 3 prefs):**
- All-at-once injection readout fixates on 1 pref, misses the "irregular schedule / no fixed-focus" constraint
  → recommendation hits a **neg** (Pomodoro).
- **Per-fact injection readout captures ALL facts incl. the constraint → reasoning AVOIDS neg and matches the
  pure-text baseline (near-identical output).**

**Full measured run (16 questions, 6 users, GPU; natural era; JEV judge; four base sizes):**

| arm | 0.8B | 4B | 9B | 27B |
|---|---|---|---|---|
| N no memory | 0.421 | 0.451 | 0.476 | 0.486 |
| P all-facts text | 0.462 | 0.469 | 0.474 | **0.556** |
| RAG retrieved text | **0.506** | 0.438 | 0.403 | 0.526 |
| H native, single-shot | 0.372 | 0.480 | **0.514** | 0.498 |
| H two-stage (this section) | 0.393 | 0.448 | 0.411 | 0.463 |

**Honest reading — the single-question advantage did NOT generalize, and no mechanism is distinguishable:**
1. **No memory arm significantly beats no-memory at any of the four sizes** (|Δ|≤0.09, s.e.≥0.04, n=16,
   per-question σ≈0.2). The bottleneck is this split's **discriminative power**, not the mechanism —
   N alone rises 0.421→0.486 with capacity, squeezing the headroom.
2. The two-stage readout **does not beat text at any size** (−0.003…−0.064 vs N). The n=1 micro-result was a
   **mechanism probe, not a score** — its projection is **refuted by measurement**.
3. **Capacity hypothesis not confirmed**: native H does flip from negative at 0.8B (−0.049) to positive at
   4B/9B (+0.029/+0.038), directionally consistent with "native injection needs capacity", but never significant;
   and at 27B the strongest arm is **all-text P (0.556)**, arguing large bases favour text, not native injection.

Next step is methodological: a **more discriminative criterion** (e.g. neg-hit rate — did it recommend a
user-rejected option — or the subset where the preference must change the answer) rather than more arms.
Evidence: `results/mempal/mempal_combo_*_{judged,results}.json`.

## 4. Selectivity-regime advantage — *template-era preliminary, pending natural-era re-validation*
Collection/催收 over-context history (W96/W97, **template era**): 27B combined H 0.535 > P 0.370 > N 0.361,
H>P 9/10; 0.8B H>P 10/10. This is the scenario where native selective injection is expected to win (text-state
must stuff everything). **These numbers predate de-templating and must be re-run in the natural era before being
cited as definitive.**

## 5. Negative results (see CLAIMS.md)
- Reasoning-bound native injection not semantically used (W103, template era, H==Hwrong 64/64).
- Fixation needs eviction (W93); attention gating ineffective.
- Forced irrelevant injection pollutes without threshold (W110).
- Injection-scale amplification degenerates (W111).
- Residual ~12-point prior resistance on conflict (0.80 vs clean 0.92, 4B), stubborn to scale.
- Base won't/can't answer → memory can't rescue (LongMemEval 0.8B refusal, W99).

## 6. Per-customer weight memory (brain-style) — separate paradigm, synthetic mechanism evidence
A deliberate, owner-approved exploration of the *opposite* commitment to the external B-KV line: customer
knowledge in **per-customer LoRA weights** (frozen Qwen3.5-9B base, rank-16 q/v LoRA), recall by generation
with **no external injection**, trained with the grouped-counterfactual use-objective. Full characterization
in [`WEIGHT_MEMORY.md`](WEIGHT_MEMORY.md); evidence `results/weightmem/`. Highlights: weight memory used
correctly (choice 8/8 vs base 4/8, flips 4/4); capacity ≥150 facts; migrates to new action words (6/6);
context-discriminates (1.0); revisable (8% old intrusion); **replay eliminates continual forgetting**
(no-replay oldest-batch retention 0.87→0.67 as scale grows; replay 1.0). Costs: few-shot *use* slow;
**audit-by-asking enumeration recall 0** (usable-but-not-reportable); not deletable/rollback-able by design.
**Synthetic customers only — mechanism evidence, not capability.** Related: Piccolo (general decision engine
on the same base; this line contributes the memory side).

## Deployment rule of thumb (honest, scoped)
Native K/V injection is a validated **writable fact-memory** mechanism (write/fuzzy-recall/overwrite/inject),
competitive on recall tasks and context-efficient. It is **not** a universal win over text/RAG: on small clean
fact sets text is competitive, and on reasoning-bound tasks text/RAG or a larger base is better. The clearest
native advantage (selectivity over long noisy history) is **template-era preliminary**, pending natural-era
re-validation.