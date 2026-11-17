# Experiments — natural-era results, arm comparisons, honest boundaries

**English** | [中文](EXPERIMENTS.zh.md)

Companion to [`CLAIMS.md`](CLAIMS.md). Raw 4B MAB result JSONs in [`results/mab/`](results/mab/).

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

## 3. Reasoning-bound preference recommendation (Mem-PAL, natural era, partial)
Mem-PAL = recall a user's long-term preferences from ~30 sessions, recommend for a vague query (reasoning-bound).
Natural-era baseline (0.8B, 8/16 queries, current recall): **N 0.335 / P 0.455 / H 0.305 / RAG 0.425 / SUM 0.334**.
**Native H still trails RAG/P** on this reasoning-bound task. (Full 16-query natural-era run + a semantic-recall
variant were started and paused; see W113 — semantic recall helped a tiny smoke test but *not* the first real
query, so it is **not** a validated fix.)

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

## Deployment rule of thumb (honest, scoped)
Native K/V injection is a validated **writable fact-memory** mechanism (write/fuzzy-recall/overwrite/inject),
competitive on recall tasks and context-efficient. It is **not** a universal win over text/RAG: on small clean
fact sets text is competitive, and on reasoning-bound tasks text/RAG or a larger base is better. The clearest
native advantage (selectivity over long noisy history) is **template-era preliminary**, pending natural-era
re-validation.