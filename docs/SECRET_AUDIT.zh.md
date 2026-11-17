# 密钥 / IP 清扫审计（M7 #3）

[English](SECRET_AUDIT.md) | **中文**

日期：2026-11-17。范围：`huamao_bkv/`、`benchmarks/`、`scripts/`、`tests/`、根目录 `*.md/*.toml`。

## 方法
`grep -rnE` 扫描：个人路径（`/home/fredgu`、`workspace-data`、`/root/autodl`）；凭证
（`api_key=`、`password`、`secret=`、长 token 字面量、`sk-…`、PEM 私钥）；基础设施痕迹
（`seetacloud`、`proxy_host`、`ssh_port`、`developer-token`）；IPv4 字面量；服务端点
（`maas.aliyuncs`、`token-plan`、`QWEN_TOKEN`）。

## 结果
- 硬编码个人路径：**0**
- 凭证 / 密钥 / token：**0**
- AutoDL / SSH / 代理 / 实例痕迹：**0**
- IPv4 字面量：**0**
- 服务端点 / key 环境变量残留：**0**（带阿里云端点的旧 `llm.py` 已移出范围至 seed 备份，不属于本发布）

## 预防性控制
- `.gitignore` 排除权重（`*.pt/*.safetensors/*.bin`）、`data/`、`*.parquet`、`.env`、`*.key`、
  `*.csv`、`*.jsonl`、`runs/`、`.venv/`。
- 底座模型权重与训练适配器**不提交**；获取方式为下载 / 自备。
- 超范围 seed 文件（native_memory、风控 harness、旧测试、`sitecustomize.py`、UCI 下载脚本）移 NFS 备份，不发布。

## 残余 / 待办
- 适配器许可：**已解决**——底座 Qwen3.5（0.8B/4B）为 Apache-2.0（允许衍生再分发）。
  训练适配器权重按决定**不发**；原理＋配方见 `docs/ADAPTER.zh.md`。
- 任何公开 push 前 / 新增文件后，重跑本审计。