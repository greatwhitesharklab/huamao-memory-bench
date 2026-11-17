#!/usr/bin/env python3
"""Download MemoryAgentBench Conflict_Resolution (MIT, ai-hyz/MemoryAgentBench) into data/mab/
(gitignored) and prepare sh6k.json / sh6k_clean.json. Data is NOT committed to the repo.

Requires: huggingface_hub, pandas, pyarrow.  Run:  python scripts/download_mab.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / 'data' / 'mab'


def main():
    from huggingface_hub import hf_hub_download
    DATADIR.mkdir(parents=True, exist_ok=True)
    print('downloading ai-hyz/MemoryAgentBench Conflict_Resolution (MIT) ...', flush=True)
    p = hf_hub_download('ai-hyz/MemoryAgentBench',
                        'data/Conflict_Resolution-00000-of-00001.parquet',
                        repo_type='dataset', local_dir=str(DATADIR))
    print('parquet ->', p, flush=True)
    subprocess.run([sys.executable, str(ROOT / 'benchmarks' / 'mab' / 'prepare_data.py'),
                    '--parquet', str(p), '--outdir', str(DATADIR)], check=True)
    print('\nDone. Run the benchmark, e.g.:')
    print('  python benchmarks/mab/run.py --model /path/Qwen3.5-0.8B --adapter /path/adapter.pt \\')
    print('      --data data/mab/sh6k.json --ret-layer 11 --ret-head 2 --n 40')


if __name__ == '__main__':
    main()