"""M6 H 档：B-KV 原生记忆引擎（M2/M5 交付形态，非旧 engram 接口）。

从 kv_realcorpus_eval.py 提取复用：冻结模型 + full-attn 层 K/V 缓存 + 共享残差适配器
+ ent_mean 检索（ret-layer/ret-head）+ K/V 注入生成。写历史事实→缓存 K/V；当前话语→
检索相关事实→注入 K/V→生成 agent 回应。这是「真正的花猫」，H 档据此对照 N/P。
"""
from __future__ import annotations
import importlib
import types
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BKVMemory:
    def __init__(self, model_path, adapter_path=None, layers=(11, 15, 19, 23),
                 ret_layer=11, ret_head=2, dtype='float32', device='cpu',
                 injection_scale=1.0, overwrite_conflicts=True):
        self.device = device
        self.layers = list(layers)
        self.ret_layer = ret_layer
        self.ret_head = ret_head
        self.injection_scale = injection_scale   # <1 轻注入（记忆当 hint 非 command）
        dt = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=dt,
            attn_implementation='sdpa').to(device)
        self.model.eval()
        torch.set_default_device(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        tcfg = self.model.config.get_text_config()
        self.n_heads, self.head_dim, self.n_kv = (tcfg.num_attention_heads,
                                                  tcfg.head_dim, tcfg.num_key_value_heads)
        self.kv_head = ret_head // (self.n_heads // self.n_kv)
        full_layers = [i for i, t in enumerate(tcfg.layer_types) if t == 'full_attention']
        attn = self.model.layers[ret_layer].self_attn if hasattr(self.model, 'layers') \
            else self.model.model.layers[ret_layer].self_attn
        self.Wq = attn.q_proj.weight.detach().float()
        self.Wk = attn.k_proj.weight.detach().float()
        self.q_norm, self.k_norm = attn.q_norm, attn.k_norm
        self._layers_mod = self.model.model.layers
        self.captured = {}
        self._layers_mod[ret_layer].register_forward_hook(
            lambda m, i_, o: self.captured.__setitem__('h', i_[0].detach()))
        self.state = {'extra': None, 'capture': False, 'kv': {}}
        self._patch_forward()
        # adapter
        if adapter_path:
            blob = torch.load(adapter_path, map_location='cpu', weights_only=False)
            self.A = {li: {t: blob[li][t].float().to(device) for t in ('K', 'V')}
                      for li in self.layers}
        else:
            self.A = {li: {t: torch.zeros(self.head_dim, self.head_dim, device=device)
                           for t in ('K', 'V')} for li in self.layers}
        self.cache = []      # [{li: (K,V)}]
        self.keys = []       # retrieval keys [T, head_dim]
        self.meta = []       # [(entity, attribute, value)]
        self.Kdoc = None
        self.msk = None
        # 冲突消解（W108）：写入覆盖开关。主键＝三元组前两元 (entity, attribute)，
        # 结构化字段、机制零文本解析；同主键新值原地顶替旧行，旧行进 superseded
        # 审计日志（外部可检视可回滚，满足客户状态可审计硬约束）。
        # 默认开（W108 验证后翻默认）；事件日志多值/历史口径可比的调用方显式传 False。
        self.overwrite_conflicts = overwrite_conflicts
        self.superseded = []

    def _patch_forward(self):
        state, layers = self.state, self.layers
        head_dim = self.head_dim
        track_layer = self.ret_layer          # 策略3：在此层追踪记忆注意力
        n_rep = self.n_heads // self.n_kv     # GQA 扩张倍数

        def patched_forward(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=None, **kwargs):
            li = self.layer_idx
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            if state['capture'] and li in layers:
                state['kv'][li] = [None, None, value_states.detach().clone()]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if state['capture'] and li in layers:
                state['kv'][li][1] = key_states.detach().clone()
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, li)
            extra = state['extra'].get(li) if state['extra'] else None
            if extra is not None:
                ek, ev = extra
                Lextra = ek.shape[2]
                key_states = torch.cat([ek.to(key_states.dtype), key_states], dim=2)
                value_states = torch.cat([ev.to(value_states.dtype), value_states], dim=2)
                Lq, Lk = query_states.shape[2], key_states.shape[2]
                if attention_mask is None:
                    m = torch.zeros(Lq, Lk, dtype=query_states.dtype, device=query_states.device)
                    causal = torch.triu(torch.full((Lq, Lq), torch.finfo(query_states.dtype).min,
                                                   dtype=query_states.dtype, device=query_states.device), diagonal=1)
                    m[:, Lextra:] = causal
                    attention_mask = m.unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 4:
                    pad = torch.zeros((*attention_mask.shape[:-1], Lextra),
                                      dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([pad, attention_mask], dim=-1)
                if li == track_layer:   # 策略3：末位 query 对记忆位的平均注意力
                    q = query_states[:, :, -1:, :].float()
                    k = key_states.repeat_interleave(n_rep, dim=1).float()
                    probs = ((q @ k.transpose(-1, -2)) * self.scaling).softmax(-1)
                    state['mem_attn'] = probs[..., :Lextra].sum(-1).mean().item()
            attention_interface = _mod.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, _mod.eager_attention_forward)
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=self.scaling, **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            return self.o_proj(attn_output), attn_weights

        attn_ret = self._layers_mod[self.ret_layer].self_attn
        _mod = importlib.import_module(type(attn_ret).__module__)
        globals()['apply_rotary_pos_emb'] = _mod.apply_rotary_pos_emb
        full_layers = [i for i, t in enumerate(self.model.config.get_text_config().layer_types)
                       if t == 'full_attention']
        for li in full_layers:
            self._layers_mod[li].self_attn.forward = types.MethodType(
                patched_forward, self._layers_mod[li].self_attn)

    def _find_span(self, dids, tids):
        for i in range(len(dids) - len(tids) + 1):
            if dids[i:i + len(tids)] == tids:
                return i, i + len(tids)
        return None

    def _overwrite_slot(self, entity, attribute):
        """覆盖开时查同主键 (entity, attribute) 的既有行号；无则 None。
        主键是写入 API 的结构化字段（如数据库唯一约束），机制不解析任何文本。"""
        if not self.overwrite_conflicts:
            return None
        for i, m in enumerate(self.meta):
            if m[0] == entity and m[1] == attribute:
                return i
        return None

    def _store_row(self, entity, attribute, value, row, key):
        """落行：覆盖开且同主键已存在→原地顶替＋旧行入审计；否则追加。返回被顶替行号或 None。"""
        slot = self._overwrite_slot(entity, attribute)
        new_meta = (entity, attribute, value)
        if slot is not None:
            self.superseded.append({'key': (entity, attribute),
                                    'old_meta': self.meta[slot],
                                    'old_keys': self.keys[slot],
                                    'old_cache': self.cache[slot],
                                    'new_meta': new_meta})
            self.cache[slot] = row
            self.keys[slot] = key
            self.meta[slot] = new_meta
        else:
            self.cache.append(row)
            self.keys.append(key)
            self.meta.append(new_meta)
        return slot

    @torch.no_grad()
    def add_fact(self, entity, attribute, value, doc=None, span_mode='char'):
        """默认自然句编码（W109 翻默认）：span_mode='char' 字符偏移→token span，语言无关、
        无硬模板，合"编码模型原生/存储与语言无关"。英文值首 token 带前导空格致精确 token
        匹配 0%（W75），字符偏移法 455/455 全纯（W109）。
        **doc 应传事实原句**（干净自然句，W109 0.92 的口径）；doc=None 时按三字段重构
        `{entity} {attribute} {value}` 兜底——span 可切但句子可能不干净，质量低于原句。
        span_mode='token'＋中文模板 doc 为旧契约（W29），供未验证自然句的调用方显式回退。"""
        doc = doc or f'{entity} {attribute} {value}'
        if span_mode == 'char':
            c0 = doc.find(value)
            if c0 < 0:
                return False
            c1 = c0 + len(value)
            enc = self.tok(doc, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc.input_ids
            ts = [i for i, (a, b) in enumerate(enc.offset_mapping) if b > a and b > c0 and a < c1]
            if not ts:
                return False
            sp = (min(ts), max(ts) + 1)
        else:
            ids = self.tok(doc, add_special_tokens=False).input_ids
            sp = self._find_span(ids, self.tok(value, add_special_tokens=False).input_ids)
            if sp is None:
                return False
        self.state['capture'] = True
        self.state['kv'] = {}
        self.model(input_ids=torch.tensor([ids]), use_cache=False)
        self.state['capture'] = False
        h = self.captured['h'][0].float()
        key = torch.nn.functional.normalize(
            self.k_norm((h @ self.Wk.t()).view(-1, self.n_kv, self.head_dim)),
            dim=-1)[:, self.kv_head, :]
        s, e = sp
        row = {li: (self.state['kv'][li][1][:, :, s:e, :].clone(),
                    self.state['kv'][li][2][:, :, s:e, :].clone())
               for li in self.layers}
        self._store_row(entity, attribute, value, row, key)
        self._rebuild_index()
        return True

    def add_fact_template(self, entity, attribute, value):
        """旧中文模板契约（W29）显式回退口：`资料：X的Y是Z。`＋精确 token span。
        仅供**尚未验证自然句编码**的调用方（风控/生产域，0.8B 适配器渲染口径未定）临时使用；
        默认 add_fact 已是自然句（W109）。验证后应迁回 add_fact。"""
        return self.add_fact(entity, attribute, value,
                             doc=f'资料：{entity}的{attribute}是{value}。', span_mode='token')

    def reset(self):
        """清缓存（复用模型，场景间隔离 H 档写入）。"""
        self.cache = []; self.keys = []; self.meta = []
        self.Kdoc = None; self.msk = None
        self.superseded = []
        self.state['extra'] = None; self.state['kv'] = {}

    def _rebuild_index(self):
        width = max(k.shape[0] for k in self.keys)
        self.Kdoc = torch.zeros(len(self.keys), width, self.head_dim)
        self.msk = torch.zeros(len(self.keys), width, dtype=torch.bool)
        for i, k in enumerate(self.keys):
            self.Kdoc[i, :k.shape[0]] = k
            self.msk[i, :k.shape[0]] = True

    @torch.no_grad()
    def _qvec(self, text):
        ids = self.tok(text, add_special_tokens=False).input_ids
        self.model(input_ids=torch.tensor([ids]), use_cache=False)
        h = self.captured['h'][0].float()
        q = h @ self.Wq.t()
        if q.shape[-1] == 2 * self.n_heads * self.head_dim:
            q = q[:, :self.n_heads * self.head_dim]
        q = torch.nn.functional.normalize(
            self.q_norm(q.view(-1, self.n_heads, self.head_dim))[:, self.ret_head], dim=-1)
        return q, ids

    @torch.no_grad()
    def recall(self, query, topk=3, mode='mean_all'):
        """mode='mean_all'（默认，原行为）对 query 全位置取 mean；
        mode='ent_mean'（M1 精化，W53/W54）只对问句中实体名位置取 mean，避开
        what/is/for 等通用位置稀释。实体 span 用实体库存匹配（非 oracle）。"""
        if not self.keys:
            return []
        q, ids = self._qvec(query)
        S = torch.einsum('td,nkd->tnk', q, self.Kdoc).masked_fill(
            ~self.msk.unsqueeze(0), -1e9).amax(-1)      # [Tq, n_facts]
        if mode == 'ent_mean':
            sp = self._query_entity_span(query)
            sc = S[sp[0]:sp[1]].mean(0) if sp is not None else S.mean(0)
        else:
            sc = S.mean(0)
        order = torch.argsort(-sc)[:topk].tolist()
        return [(self.meta[i], float(sc[i])) for i in order]

    def _query_entity_span(self, query):
        """实体库存匹配：在问句中找已存实体名（取最长匹配），char-offset→token span。
        非 oracle——只用已存实体清单做实体链接，不偷看答案。找不到返回 None（退回 mean_all）。"""
        ql = query.lower()
        best = None
        for (e, _a, _v) in self.meta:
            c0 = ql.find(e.lower())
            if c0 >= 0 and (best is None or len(e) > best[2]):
                best = (c0, c0 + len(e), len(e))
        if best is None:
            return None
        offs = self.tok(query, return_offsets_mapping=True,
                        add_special_tokens=False).offset_mapping
        c0, c1 = best[0], best[1]
        ts = [i for i, (a, b) in enumerate(offs) if b > a and b > c0 and a < c1]
        if not ts:
            return None
        return (min(ts), max(ts) + 1)

    def _build_extra(self, idxs):
        out = {}
        s = self.injection_scale
        for li in self.layers:
            K = torch.cat([self.cache[i][li][0] for i in idxs], dim=2)
            V = torch.cat([self.cache[i][li][1] for i in idxs], dim=2)
            out[li] = (K + s * (K @ self.A[li]['K'].to(K.dtype)),
                       V + s * (V @ self.A[li]['V'].to(V.dtype)))
        return out

    @torch.no_grad()
    def generate(self, prompt_text, recall_idxs=None, max_new=64, repetition_penalty=1.3,
                 evict_after=None, evict_mode='time', attn_threshold=0.05, repeat_n=4,
                 fact_values=None, skip_special_tokens=True):
        """记忆驱逐（脑启发 transient 激活）。cache 只存上下文 K/V，记忆每步重拼到注意力前部；
        驱逐=停止重拼（清 extra）。prefill 记忆影响上下文编码，驱逐后影响 baked 进上下文 K/V。
        evict_mode='time': evict_after=None 永不（旧H）/0 prefill后（策略1）/N 前N步后（策略2）。
        evict_mode='attn': 策略3 注意力门控——末位 query 对记忆位平均注意力 < attn_threshold 即驱逐
        （模型不再用记忆就撤，比定时自适应；W93 发现 fixation 即持续注意，循环无效）。
        evict_mode='repeat': 策略4 输出状态自适应——检测最近 repeat_n-gram 在输出中重复出现
        （fixation 签名）即驱逐。治固定步驱逐抓不住随机 onset 的问题（W100 延迟方差）。"""
        self.state['extra'] = self._build_extra(recall_idxs) if recall_idxs else None
        self.state['mem_attn'] = None
        ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        cur = torch.tensor([ids]); past = None; out = []
        # readout 模式：预计算事实值 token 序列，输出一旦表达某事实即驱逐（fixation 前）
        fact_seqs = []
        if evict_mode == 'readout' and fact_values:
            for fv in fact_values:
                ft = self.tok(fv, add_special_tokens=False).input_ids
                if len(ft) >= 2:
                    fact_seqs.append(ft)
        for i in range(max_new):
            stp = self.model(input_ids=cur, past_key_values=past, use_cache=True)
            past = stp.past_key_values
            logits = stp.logits[:, -1, :].float()
            if repetition_penalty != 1.0 and out:
                for prev in set(out):
                    logits[0, prev] = (logits[0, prev] / repetition_penalty
                                       if logits[0, prev] > 0
                                       else logits[0, prev] * repetition_penalty)
            t = logits.argmax(-1).item()
            if self.state['extra'] is not None:
                if evict_mode == 'attn':
                    ma = self.state.get('mem_attn')
                    if ma is not None and ma < attn_threshold:
                        self.state['extra'] = None   # 注意力门控驱逐
                elif evict_mode == 'repeat':
                    # 输出状态自适应：最近 repeat_n-gram 曾在输出中出现→fixation→驱逐
                    # （W101 发现太晚：fixation 已 baked 进上下文，驱逐止不住）
                    if len(out) >= repeat_n * 2:
                        last_ng = tuple(out[-repeat_n:])
                        prev = out[:-repeat_n]
                        if any(tuple(prev[j:j + repeat_n]) == last_ng
                               for j in range(len(prev) - repeat_n + 1)):
                            self.state['extra'] = None
                elif evict_mode == 'readout':
                    # 读出即驱逐：输出表达某召回事实（token 子序列匹配）即撤——fixation 前
                    for ft in fact_seqs:
                        if len(out) >= len(ft) and out[-len(ft):] == ft:
                            self.state['extra'] = None
                            break
                elif evict_after is not None and i == evict_after:
                    self.state['extra'] = None       # 定时驱逐
            if t == self.tok.eos_token_id:
                break
            out.append(t); cur = torch.tensor([[t]])
        self.state['extra'] = None
        return self.tok.decode(out, skip_special_tokens=skip_special_tokens).strip()