"""huamao_bkv — B-KV native writable memory for a frozen small language model.

Public surface:
    BKVMemory  — frozen base + full-attn-layer K/V cache + shared residual adapter
                 + ent_mean retrieval + K/V injection generation, with conflict
                 overwrite (structured primary key + audit) and natural-sentence
                 encoding (char-offset value span).

The base model weights are NOT bundled; point ``model_path`` at a local Qwen3.5
checkpoint and ``adapter_path`` at a trained shared adapter (see README).
"""
from .memory import BKVMemory

__all__ = ["BKVMemory"]
__version__ = "0.1.0"