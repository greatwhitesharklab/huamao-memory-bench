#!/usr/bin/env python3
"""Prepare MemoryAgentBench Conflict_Resolution single-hop-6k (sh_6k) for the B-KV benchmark.

Reads the MAB parquet (see scripts/download_mab.sh), extracts the sh_6k row (row index 4:
single-hop questions over the 6k-token fact haystack), and writes:
  sh6k.json        {context, facts, qa}  — full conflict set (facts include updated/counterfactual duplicates)
  sh6k_clean.json  non-conflict subset (support-entity appears exactly once) for recall-capacity probing

MAB is MIT-licensed (HUST-AI-HYZ/MemoryAgentBench, arXiv:2507.05257). No model weights here.
"""
from __future__ import annotations
import argparse, ast, json, re, collections
from pathlib import Path


def parse_fact(line):
    """(entity, value) from a numbered MAB fact line. Generic templates, no corpus-specific words."""
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


STOP = set('what which who whom whose is are was were the a an of in on at to for by with as that this it its his her their and or do does did'.split())
def toks(s): return set(w for w in re.findall(r"[A-Za-zÀ-ÿ0-9']+", s.lower()) if w not in STOP and len(w) > 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', required=True, help='path to Conflict_Resolution.parquet')
    p.add_argument('--outdir', default='data/mab')
    p.add_argument('--row', type=int, default=4, help='sh_6k row index (single-hop, 6k context)')
    a = p.parse_args()
    import pandas as pd
    df = pd.read_parquet(a.parquet)
    row = df.iloc[a.row]
    context = row['context']
    questions = row['questions']; answers = row['answers']
    if isinstance(questions, str): questions = ast.literal_eval(questions)
    facts = [ln for ln in context.split('\n') if re.match(r'^\d+\.', ln.strip())]
    qa = []
    for q, ans in zip(questions, answers):
        av = ans[0] if hasattr(ans, '__len__') and not isinstance(ans, str) else ans
        av = str(av)
        support = [f for f in facts if av.lower() in f.lower()]
        qa.append({'q': q, 'a': av, 'support': support})
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    json.dump({'context': context, 'facts': facts, 'qa': qa},
              open(out / 'sh6k.json', 'w'), ensure_ascii=False, indent=1)
    # non-conflict subset: support-entity appears exactly once among facts
    ent_of = {}; cnt = collections.Counter()
    for ln in facts:
        e, v = parse_fact(ln)
        ent_of[ln] = e.strip().lower(); cnt[e.strip().lower()] += 1
    clean_facts = [ln for ln in facts if cnt[ent_of[ln]] == 1]
    clean_qa = []
    for pp in qa:
        cand = [ln for ln in facts if pp['a'].lower() in ln.lower()]
        if not cand: continue
        qt = toks(pp['q'])
        sup = max(cand, key=lambda ln: len(toks(ln) & qt))
        if cnt[ent_of[sup]] == 1: clean_qa.append(pp)
    json.dump({'context': context, 'facts': clean_facts, 'qa': clean_qa},
              open(out / 'sh6k_clean.json', 'w'), ensure_ascii=False, indent=1)
    print(f'facts={len(facts)} qa={len(qa)} -> sh6k.json')
    print(f'clean: facts={len(clean_facts)} qa={len(clean_qa)} -> sh6k_clean.json')


if __name__ == '__main__':
    main()