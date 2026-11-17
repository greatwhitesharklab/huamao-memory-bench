# huamao-bkv — 冻结小模型之外的原生可写 K/V 记忆

[English](README.md) | **中文**

**花猫（huamao）**：在一个**冻结**的小语言模型**外部**，建立可写、可覆盖、可模糊检索的记忆——
不扩大模型参数，也不扩大上下文窗口。召回时把记住的事实以 **K/V 行**注入模型注意力，由模型自身的
decode 收敛出答案。

> **状态：机制库＋可复现基准。能力结论待验证（pending）。**
> 引用任何数字前请先读 [`CLAIMS.md`](CLAIMS.md)——负结果也如实列在那里。

## 盒子里有什么
- `huamao_bkv/` — B-KV 引擎（`BKVMemory`）：冻结底座＋full-attn 层 K/V 缓存＋一个小型**共享**残差
  适配器＋`ent_mean` 模糊检索＋K/V 注入生成。
  - **冲突覆盖**：结构化主键 `(entity, attribute)`，最新值获胜，带 `superseded` 审计轨迹（可检视/可回滚）。
    机制内**零文本解析**。
  - **自然句编码**：字符偏移→token 值 span（语言无关，无硬编码模板）。
- `benchmarks/mab/` — 可复现的 **MemoryAgentBench**（Conflict_Resolution，单跳 6k）机制基准，
  **0.8B/4B CPU** 即可跑。
- `tests/` — 离线单测（无需模型权重/GPU）。

## 快速上手（0.8B CPU）
```bash
pip install -e .                 # torch + transformers
pip install pandas pyarrow huggingface_hub   # 基准额外依赖

# 1) 底座权重不随仓附带——自行下载 Qwen3.5-0.8B（ModelScope 或 HF），例如
#    modelscope download Qwen/Qwen3.5-0.8B --local_dir ./models/Qwen3.5-0.8B
# 2) 基准数据（MIT，下载而非提交进仓）：
python scripts/download_mab.py
# 3) 运行（--adapter 指向训练好的共享适配器，见下文「适配器」）：
python benchmarks/mab/run.py --model ./models/Qwen3.5-0.8B --adapter /path/adapter.pt \
    --data data/mab/sh6k.json --ret-layer 11 --ret-head 2 --n 40
```
消融开关：`--no-overwrite`（只追加不覆盖）、`--template`（旧中文模板编码）。

## 适配器与权重（许可）
- **Qwen 底座权重不再分发**——按其自身许可下载（Qwen3.5 为 Apache-2.0）。
- **共享适配器**（约 52 万参数）是让模型真正*用上*注入 K/V 的读取接口（没有它，利用率从约 95% 掉到约 17.5%）。
  **我们只发原理＋训练配方，不发训练好的权重**——见 [`docs/ADAPTER.md`](docs/ADAPTER.md)，可在你自己的底座/领域上训练。
  机制在零适配器（省略 `--adapter`）下也能跑，但注入利用率会骤降。
- 许可已核：Qwen3.5（0.8B/4B）为 **Apache-2.0**（无 Qwen 限制性条款），允许衍生适配器再分发；
  我们只发原理是**主动决定**，不是许可障碍。

## 工作原理（一段话）
写入一条事实→把它的值 span 编码成逐层 K/V 行（冻结前向）→存行＋检索键。查询时 `ent_mean` 用问句的
实体位置对存储键打分，返回 top-k；这些 K/V（加共享适配器残差）在 prefill/decode 时拼到注意力前部，模型生成。
冲突更新覆盖同一主键（保留审计）。召回**阈值**门控注入，避免无关事实污染无关回答。

## 相关工作（定位）
- **RETRO**（DeepMind）：检索增强*预训练*＋分块交叉注意力。B-KV 是事后式：把可写 K/V 注入已冻结的模型，无需预训练。
- **Engram**：n-gram 哈希查找的条件记忆（需训练）。B-KV 用模糊 K/V 检索＋注入。
- **NVIDIA arXiv:2611.05306**：模型自压缩文本态再回注（我们的「C 档」候选）。
- **基准**：LongMemEval、Mem-PAL、**MemoryAgentBench**（arXiv:2507.05257，MIT）——我们用 MAB
  Conflict_Resolution 探更新/覆盖＋反事实注入。

## 诚实的边界
记忆对**知识/召回绑定**任务有效（小模型＋记忆可逼近大模型读出）。它**买不来**多跳推理容量；
在推理绑定任务上，原生注入甚至可能完全不被语义利用。见 [`CLAIMS.md`](CLAIMS.md)。

## 许可
Apache-2.0（本仓）。基准数据：MIT（MemoryAgentBench）。底座模型权重：各自许可，不随仓附带。