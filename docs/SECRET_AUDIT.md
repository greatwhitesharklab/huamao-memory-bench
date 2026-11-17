# Secret / IP scrub audit (M7 #3)

Date: 2026-11-17. Scope: `huamao_bkv/`, `benchmarks/`, `scripts/`, `tests/`, root `*.md/*.toml/*.txt`.

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

## Preventive controls
- `.gitignore` excludes weights (`*.pt/*.safetensors/*.bin`), `data/`, `*.parquet`, `.env`, `*.key`,
  `*.csv`, `*.jsonl`, `runs/`, `.venv/`.
- Base model weights and the trained adapter are **not committed**; acquisition is by download / bring-your-own.
- Out-of-scope seed files (native_memory, risk harness, old tests, `sitecustomize.py`, UCI download script)
  moved to NFS backup, not released.

## Residual / pending
- Adapter redistribution license review (Qwen) — M7 #2, pending. Adapter is not committed meanwhile.
- Re-run this audit before any public push / after adding new files.