#!/usr/bin/env python3
"""本地判分：把 GPU 实例产出的 pred 用 JEV 判 pos/neg 对齐，算各臂均分。
结果写回 *_judged.json 与 _summary.json。"""
import sys, json
from pathlib import Path
MAIN = Path('<main-repo>/incubation/anvil-risk-workflow')
sys.path.insert(0, str(MAIN / 'experiments')); sys.path.insert(0, str(MAIN / 'core'))
from m6_bench.mempal import load_mempal
from m6_bench.run_mempal import judge_align
from native_memory.consolidation import JevDecider

RUN = Path('<runs>/run-263-mempal-combo')
DATA = Path('<runs>/run-251-corpus/Mem-PAL/input.json')
ARMS = ('N', 'P', 'RAG', 'Hent', 'Hcombo')


def main(tag):
    res = json.loads((RUN / f'mempal_combo_{tag}_results.json').read_text())
    data = load_mempal(DATA)
    decider = JevDecider()
    # 建 (user, query) -> topic
    idx = {}
    for uid, u in data.items():
        qs = u['query'] if isinstance(u['query'], list) else list(u['query'].values())
        for q in qs:
            tops = q['topics'] if isinstance(q['topics'], list) else list(q['topics'].values())
            for t in tops:
                idx[(uid, t['user_query'])] = t
    missing = 0
    for r in res:
        t = idx.get((r['user'], r['query']))
        if t is None:
            missing += 1; continue
        for c in ARMS:
            if c in r['conditions']:
                r['conditions'][c]['align'] = judge_align(decider, t, r['conditions'][c]['pred'])
    summary = {c: round(sum(r['conditions'][c]['align'] for r in res if c in r['conditions'])
                        / max(1, sum(1 for r in res if c in r['conditions'])), 3) for c in ARMS}
    (RUN / f'mempal_combo_{tag}_judged.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (RUN / f'mempal_combo_{tag}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print('缺失topic:', missing, '| n:', len(res))
    print('SUMMARY:', json.dumps(summary, ensure_ascii=False))
    for r in res:
        print(r['user'], r['query'][:20], {c: r['conditions'][c]['align'] for c in ARMS if c in r['conditions']})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'full4b_18q')