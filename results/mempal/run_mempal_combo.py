#!/usr/bin/env python3
"""Mem-PAL 全量合体跑法（W113 §14）：同 run 内对照五臂。
N 无记忆 / P 全文本 / RAG BM25会话文本 / Hent 整条注入+ent_mean单发生成(原生现状) /
Hcombo 整条注入+语义召回+逐条读出→事实作文本→推理（脑启发两段式）。
配置对齐 run-260 基线：0.8B、layers(11,15,19,23)、ret_layer=11、context_budget=8192、6用户×1样本×3题。
用法：--model/--adapter/--layers/--ret-layer/--sem-layer/--dtype/--n-users/--max-query/--out/--tag
"""
import sys, json, time, argparse
from pathlib import Path
MAIN = Path('<main-repo>/incubation/anvil-risk-workflow')
sys.path.insert(0, str(MAIN / 'experiments')); sys.path.insert(0, str(MAIN / 'core'))
sys.path.insert(0, '<runs>/run-262-mempal-sem')
from m6_bench.mempal import load_mempal, history_to_sessions, MemPALAgent, MEM_SYS
from m6_bench.run_mempal import judge_align
from native_memory.consolidation import JevDecider
from fullfact_sem_bkv import FullFactSemBKV

DATA = Path('<runs>/run-251-corpus/Mem-PAL/input.json')
PREF_DIR = Path('<runs>/run-262-mempal-sem')
READ_SYS = '你是记忆读出器。只陈述注入的这条用户事实/偏好，忠实完整，不要给建议。'
ARMS = ('N', 'P', 'RAG', 'Hent', 'Hcombo')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--adapter', required=True)
    ap.add_argument('--layers', default='11,15,19,23'); ap.add_argument('--ret-layer', type=int, default=11)
    ap.add_argument('--ret-head', type=int, default=2); ap.add_argument('--sem-layer', type=int, default=23)
    ap.add_argument('--dtype', default='float32'); ap.add_argument('--device', default='cpu'); ap.add_argument('--n-users', type=int, default=6)
    ap.add_argument('--max-query', type=int, default=1); ap.add_argument('--budget', type=int, default=8192)
    ap.add_argument('--topk', type=int, default=3); ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--no-judge', action='store_true')   # 只产 pred，判分另跑（实例不接 JEV 时）
    a = ap.parse_args()
    layers = tuple(int(x) for x in a.layers.split(','))
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    decider = None if a.no_judge else JevDecider()
    data = load_mempal(DATA)
    bkv = FullFactSemBKV(a.model, a.adapter, layers=layers, ret_layer=a.ret_layer,
                         ret_head=a.ret_head, sem_layer=a.sem_layer, dtype=a.dtype, device=a.device)
    def chat(sysc, user):
        return bkv.tok.apply_chat_template(
            [{'role': 'system', 'content': sysc}, {'role': 'user', 'content': user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    results = []
    for uid in list(data.keys())[:a.n_users]:
        u = data[uid]
        stexts = history_to_sessions(u['history'])
        pf = PREF_DIR / f'prefs_{uid}.json'
        if not pf.exists():
            print(f'[user {uid}] MISSING prefs', flush=True); continue
        prefs = json.load(open(pf))
        bkv.reset()
        for f in prefs:
            bkv.add_fact(f.get('entity', '用户'), f['attribute'], f['value'])
        print(f'[user {uid}] sessions={len(stexts)} prefs={len(prefs)} rows={len(bkv.meta)}', flush=True)
        queries = u['query'] if isinstance(u['query'], list) else list(u['query'].values())
        for qs in queries[:a.max_query]:
            topics = qs['topics'] if isinstance(qs['topics'], list) else list(qs['topics'].values())
            for t in topics:
                q = t['user_query']
                row = {'user': uid, 'query': q, 'n_prefs': len(prefs), 'conditions': {}}
                for cond in ARMS:
                    t0 = time.time()
                    print(f'  ...{cond} start', flush=True)
                    if cond == 'Hcombo':
                        idxs = bkv.semantic_idxs(q, topk=a.topk)
                        parts = []
                        for i in idxs:
                            parts.append(bkv.generate(chat(READ_SYS, q), recall_idxs=[i],
                                                      max_new=80, repetition_penalty=1.1).strip())
                        facts = '\n'.join(f'{j+1}. {p}' for j, p in enumerate(parts) if p)
                        pred = bkv.generate(chat(MEM_SYS + '\n\n用户相关事实：\n' + facts, q),
                                            recall_idxs=None, max_new=120, repetition_penalty=1.1)
                    elif cond == 'Hent':
                        rec = bkv.recall(q, topk=a.topk, mode='ent_mean')
                        idxs2 = [i for i, m in enumerate(bkv.meta) if any(m == rm for rm, _ in rec)]
                        pred = bkv.generate(MemPALAgent(t, 'H', bkv=bkv, facts=[])._prompt(),
                                            recall_idxs=idxs2 or None, max_new=120)
                    else:
                        agent = MemPALAgent(t, 'RAG' if cond == 'RAG' else ('P' if cond == 'P' else 'N'),
                                            bkv=bkv, facts=[], session_texts=stexts,
                                            context_budget=a.budget)
                        pred = agent.propose()
                    align = 0.0 if a.no_judge else judge_align(decider, t, pred)
                    row['conditions'][cond] = {'pred': pred, 'align': align,
                                               'seconds': round(time.time() - t0, 1)}
                results.append(row)
                print(json.dumps({'done': uid, 'q': q[:24],
                                  'align': {c: row['conditions'][c]['align'] for c in ARMS},
                                  'sec': {c: row['conditions'][c]['seconds'] for c in ARMS}},
                                 ensure_ascii=False), flush=True)
    (OUT / f'mempal_combo_{a.tag}_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    summary = {c: round(sum(r['conditions'][c]['align'] for r in results) / max(1, len(results)), 3) for c in ARMS}
    (OUT / f'mempal_combo_{a.tag}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({'summary': summary, 'n': len(results)}, ensure_ascii=False), flush=True)
    print('MEMPAL_COMBO_DONE', flush=True)


if __name__ == '__main__':
    main()