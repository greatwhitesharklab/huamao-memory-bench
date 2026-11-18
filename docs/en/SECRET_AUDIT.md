# Secret / IP scrub audit (M7 #3)

**English** | [中文](../zh/SECRET_AUDIT.md)

Date: 2026-11-18 (re-run for the per-customer weight-memory release). Scope: `huamao_bkv/`, `benchmarks/`,
`scripts/`, `tests/`, root `*.md/*.toml/*.txt`, plus new `docs/{en,zh}/WEIGHT_MEMORY.md` and
`results/weightmem/*.json`.

## Method
`grep -rnE` for: personal paths (`/home/fredgu`, `workspace-data`, `/root/autodl`); credentials
(`api_key=`, `password`, `secret=`, long token literals, `sk-…`, PEM private keys); infra traces
(`seetacloud`, `proxy_host`, `ssh_port`, `developer-token`); IPv4 literals; provider endpoints
(`maas.aliyuncs`, `token-plan`, `QWEN_TOKEN`).

## Result
- Hardcoded personal paths: **0**
- Credentials / keys / tokens: **0**
- AutoDL / SSH / proxy / instance traces: **0**
- IPv4 literals: **0**
- Provider endpoint / key-env remnants: **0** (the old `llm.py` with the Aliyun endpoint was moved out
  of scope to the seed backup; it is not part of this release)

- `results/weightmem/` (2026-11-18): result JSONs copied from NFS runs; the only infra trace was the remote
  model path in `suite_config.json`/`suite_summary.json`, **scrubbed to `Qwen3.5-9B`**. Re-grep of the
  directory: personal paths / credentials / AutoDL-SSH-proxy traces / IPv4 / keys = **0**. No weights shipped.

## Preventive controls
- `.gitignore` excludes weights (`*.pt/*.safetensors/*.bin`), `data/`, `*.parquet`, `.env`, `*.key`,
  `*.csv`, `*.jsonl`, `runs/`, `.venv/`.
- Base model weights and the trained adapter are **not committed**; acquisition is by download / bring-your-own.
- Out-of-scope seed files (native_memory, risk harness, old tests, `sitecustomize.py`, UCI download script)
  moved to NFS backup, not released.

## Residual / pending
- Adapter license: **resolved** — base Qwen3.5 (0.8B/4B) is Apache-2.0 (permits derivative redistribution).
  Trained adapter weights are **not shipped** by decision; principle + recipe in `ADAPTER.md`.
- Re-run this audit before any public push / after adding new files.