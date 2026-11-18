# Claims — what is validated, what is pending, what failed

**English** | [中文](../zh/CLAIMS.md)

Honesty contract (M7): **mechanism claims are validated; capability claims are pending; negative
results are reported, not hidden.** Synthetic results are mechanism evidence only; production
capability requires real-corpus validation (project discipline).

## ✅ Validated mechanism claims

| Claim | Evidence | Scope / limit |
|---|---|---|
| B-KV injection: frozen base + full-attn K/V cache + shared residual adapter; write→fuzzy recall→inject→generate | W60–W63 (e2e 92.5–95% template same-domain) | mechanism; template same-domain is the upper bound |
| Fuzzy recall via `ent_mean` (entity-position scoring, avoids function-word dilution) | W53/W54, W104 | retrieval-side only; ranking left to decode |
| **Conflict overwrite**: structured primary key `(entity, attribute)`, latest wins, `superseded` audit (rollback-able) | W108: MAB conflict k1 0.35→0.55 (4B), 0.45→0.825 (0.8B); 10 offline unit tests | exact same-(entity,relation) updates; paraphrase conflicts not detected |
| **Natural-sentence encoding**: char-offset→token value span, language-agnostic, no hardcoded template | W109: span 455/455 pure; MAB conflict 0.55→0.80, clean 0.84→0.92 (4B) | adapter must be trained on matching render |
| **Threshold gating preserves general capability** | W110: forced irrelevant injection N=1.0→H=0.8; threshold 0.27 → H=1.0 | threshold is domain-calibrated, provisional 0.27 |
| Level-skipping on **knowledge/recall-bound** tasks (small model + memory ≈ big-model readout) | W106–W111: 4B clean 0.92 ≈ 27B oracle-readout anchor 0.925 | vs readout anchor, not 27B same-task bare run; reasoning-bound NOT covered |

## ⏳ Pending capability claims (do NOT cite as capability yet)

- **Production / risk-domain capability**: pending real corpus (Huawei-cloud approval in flight;
  27B real-corpus run pending). Synthetic risk results are mechanism-only (W93–W97).
- **27B same-task bare baseline** for the level-skipping claim: pending GPU.
- **Reasoning-bound level-skipping**: not demonstrated; capacity is a hard bottleneck (memory does
  not buy multi-hop reasoning).

## ❌ Negative results (reported, not buried)

- **Reasoning-bound injection not semantically used** (W103, real UCI risk): H and Hwrong (poisoned)
  answers identical 64/64 — injection perturbs generation trajectory, content-irrelevant. Text-state
  history (P) helps; native K/V injection does not on this task.
- **Fixation** (W93): injection can drive repetitive fixation; needs eviction (timed/readout). Attention
  gating ineffective (fixation IS sustained attention).
- **Injection-scale amplification fails** (W111): scale=2.0 degenerates (garbage + repetition) and
  regresses correct answers; amplifying injection does not cleanly override prior.
- **Forced irrelevant injection pollutes** (W110): without threshold gating, unrelated answers are
  contaminated by memory entities/phrasing. Threshold gating is required.
- **Residual ~12-point prior resistance** (W109/W111): after overwrite + natural encoding, conflict
  (0.80) still trails clean (0.92) on 4B; larger base resists counterfactual injection more. Stubborn
  to scale; likely a capacity/fusion tradeoff.
- **Bigger base ≠ better on conflict** (W106): 4B prior stronger than 0.8B → more resistance before
  overwrite; capacity helps anti-dilution, not counterfactual acceptance.

## Per-customer weight memory — separate paradigm (synthetic mechanism only)

Distinct from the external B-KV line above: knowledge in per-customer LoRA weights, not auditable rows.
Owner-approved relaxation; accepted costs: not deletable / not rollback-able / forgets / self-report ≠ state.
Details + evidence: [`WEIGHT_MEMORY.md`](WEIGHT_MEMORY.md), `results/weightmem/`.

| Claim | Evidence (synthetic, 9B) | Scope / limit |
|---|---|---|
| Per-customer LoRA stores & **uses** preferences (no external injection) | choice 8/8 vs base 4/8; flips 4/4 | mechanism; arbitrary-label synthetic customers |
| Capacity ≥150 facts; migrates to new action words; context-discriminates; revisable | 150→1.0; migration 6/6; disc 1.0; old intrusion 8% | rank16, ≤150 facts, one size |
| Replay eliminates continual forgetting | no-replay oldest 0.87→0.67; replay 1.0 | this scale only |
| **Costs:** few-shot *use* slow; **audit-by-asking enumeration recall 0** (usable-but-not-reportable) | 4 exposures choice 0.5; open-ended recall 0.0 | reported, not buried |

**Not a capability claim.** Real-corpus validation pending. Do not cite as production memory capability.

## Benchmark numbers (mechanism-level, MAB real data, MIT)

4B, natural encoding + overwrite, ent_mean top-1, CPU:
- clean recall (n=25): **0.92** (template 0.84)
- conflict (n=40): **0.80** (template 0.55; append-only 0.35)

0.8B, natural encoding + overwrite, ent_mean top-1, CPU (reproduced via `benchmarks/mab/run.py`):
- clean recall (n=25): **0.92** (template+overwrite was 0.88)
- conflict (n=40): **0.825**

**Arm comparison, MAB clean (0.8B, natural, n=15):** N 0.20 / H native **0.933** / RAG text **1.00** / P text **1.00**.
→ On a small clean single-hop fact set, **text is as good or slightly better than native injection**; native is
competitive but does not beat text here. Native's edge is selectivity/writability/context-efficiency, not this scenario.

**Mem-PAL reasoning-bound (natural, 16 questions, 6 users, GPU; measured):**

| arm | 0.8B | 4B |
|---|---|---|
| N no memory | 0.421 | 0.451 |
| P all text | 0.462 | 0.469 |
| RAG retrieved text | **0.506** | 0.438 |
| H native single-shot | 0.372 | **0.480** |
| H two-stage readout→reason | 0.393 | 0.448 |

**Honest reading:** at 4B **no memory arm significantly beats no-memory** (best +0.029, s.e. 0.053) — this
16-question split is weakly discriminative (N alone 0.45). The two-stage fix, validated on a single question,
**does not beat text at either size** — that n=1 projection is **refuted by measurement**. Only directional:
native injection improves with capacity (+0.107) while text RAG degrades (−0.068), consistent with
"memory benefit = task type × base capacity" but **not yet significant**.
Evidence: `results/mempal/`.

Limits: MAB sh_6k single-hop; n=40/25/15; substring_exact_match; conflict set is 70% conflict groups.
Selectivity-regime advantage (催收 H>P) is **template-era preliminary**, pending natural-era re-validation.