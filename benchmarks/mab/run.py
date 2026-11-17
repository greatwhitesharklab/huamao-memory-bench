#!/usr/bin/env python3
"""B-KV × MemoryAgentBench (sh_6k) reproducible mechanism benchmark — 0.8B/4B CPU.

Demonstrates: write (natural-sentence encoding + conflict overwrite) -> ent_mean recall
-> top-k K/V injection -> generation, scored by substring_exact_match.

No hardcoded paths: pass --model / --adapter / --data. Base weights are not bundled.
Example (0.8B CPU):
    python benchmarks/mab/run.py --model /path/Qwen3.5-0.8B --adapter /path/adapter.pt \
        --data data/mab/sh6k.json --ret-layer 11 --ret-head 2 --n 40
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for huamao_bkv
from huamao_bkv import BKVMemory


def parse_fact(line):
    s = re.sub(r'^\d+\.\s*', '', line.strip()).rstrip('.').strip()
    m = re.match(r'^The univeristy where (.+?) was educated is (.+)$', s, re.I)
    if m: return m.group(1), m.group(2)
    m = re.match(r'^The .+? of (.+?) (?:is|was) (.+)$', s, re.I)
    if m: return m.group(1), m.group(2)
    m = re.match(r'^(.+?) (?:is|are|was|were) (.+)$', s, re.I)
    if m:
        ent, rest = m.group(1), m.group(2)
        vm = re.search(r'(?:to|of|for|in|by|at|with) (.+)$', rest, re.I)
        return ent, (vm.group(1) if vm else rest)
    m = re.search(r'^(.*?)\b(?:of|in|by|to|for|at|with)\b(.+)$', s, re.I)
    if m and m.group(1).strip(): return m.group(1).strip(), m.group(2).strip()
    w = s.split(); return ' '.join(w[:-1]), w[-1]


def relation_of(line, e, v):
    """Relation = original sentence minus entity minus value (generic string subtraction)."""
    s = re.sub(r'^\d+\.\s*', '', line.strip())
    return ' '.join(s.replace(e, '', 1).replace(v, '', 1).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', default=None)
    p.add_argument('--data', required=True, help='sh6k.json or sh6k_clean.json')
    p.add_argument('--ret-layer', type=int, default=11)
    p.add_argument('--ret-head', type=int, default=2)
    p.add_argument('--layers', default='11,15,19,23')
    p.add_argument('--dtype', default='float32')
    p.add_argument('--n', type=int, default=40)
    p.add_argument('--topk', type=int, default=1)
    p.add_argument('--max-new', type=int, default=20)
    p.add_argument('--no-overwrite', action='store_true', help='ablation: append instead of overwrite')
    p.add_argument('--template', action='store_true', help='ablation: legacy Chinese-template encoding')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    layers = tuple(int(x) for x in a.layers.split(','))
    d = json.load(open(a.data)); facts = d['facts']; qa = d['qa'][:a.n]
    print(f'loading model {a.model} ...', flush=True)
    bkv = BKVMemory(a.model, a.adapter, layers=layers, ret_layer=a.ret_layer,
                    ret_head=a.ret_head, dtype=a.dtype, device='cpu',
                    overwrite_conflicts=not a.no_overwrite)
    tok = bkv.tok; t0 = time.time()
    for line in facts:
        e, v = parse_fact(line)
        if not (e and v): continue
        rel = relation_of(line, e, v)
        if a.template:
            bkv.add_fact_template(e, rel, v)
        else:
            doc = re.sub(r'^\d+\.\s*', '', line.strip())
            bkv.add_fact(e, rel, v, doc=doc, span_mode='char')
    print(f'write {len(bkv.meta)} rows (superseded {len(bkv.superseded)}) in {round(time.time()-t0,1)}s', flush=True)
    m2i = {}
    for i, m in enumerate(bkv.meta): m2i.setdefault(m, []).append(i)
    def prompt_for(q):
        return tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant. Answer concisely.'},
            {'role':'user','content':q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ok = 0; rows = []
    for j, pp in enumerate(qa):
        q, ans = pp['q'], pp['a']; pr = prompt_for(q)
        rec = bkv.recall(q, topk=a.topk, mode='ent_mean')
        used=set(); idxs=[]
        for rm, s in rec:
            for i in m2i.get(rm, []):
                if i not in used: used.add(i); idxs.append(i); break
        gen = bkv.generate(pr, recall_idxs=idxs or None, max_new=a.max_new, repetition_penalty=1.1)
        hit = ans.lower() in gen.lower(); ok += int(hit)
        rows.append({'q':q,'a':ans,'hit':hit,'gen':gen[:120]})
        if j+1<=3 or (j+1)%10==0:
            print(f'[{j+1}] hit={int(hit)} | {q[:40]} | {gen[:50]}', flush=True)
    acc = ok/len(qa)
    print(f'\n=== {Path(a.data).name} topk={a.topk} overwrite={not a.no_overwrite} '
          f'enc={"template" if a.template else "natural"}: acc={ok}/{len(qa)}={acc:.3f} ===')
    if a.out:
        json.dump({'n':len(qa),'topk':a.topk,'acc':round(acc,3),'hits':ok,
                   'overwrite':not a.no_overwrite,'encoding':'template' if a.template else 'natural',
                   'rows_active':len(bkv.meta),'superseded':len(bkv.superseded),'detail':rows},
                  open(a.out,'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()