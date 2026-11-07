"""Routing headroom in a scored run: per-group scores, effort and size gains by task type, and the oracle router.

The oracle picks, for each question, the cheapest configuration among those with the question's best score
(RouterBench's oracle); it bounds what any router can reach on these responses.

    python baseline/analyze.py --run pilot
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trapi  # noqa: E402
from run import EFFORTS, MODELS  # noqa: E402

CONFIGS = [(m, e) for m in MODELS for e in EFFORTS]
# Task type used when the groups were chosen: calculation-heavy vs knowledge-heavy.
TYPE = {'mmlu_pro/math': 'calc', 'mmlu_pro/physics': 'calc', 'supergpqa/Physics': 'calc',
        'supergpqa/Mathematics': 'calc', 'supergpqa/Chemistry': 'calc', 'livebench/math': 'calc',
        'mmlu_pro/health': 'knowledge', 'mmlu_pro/law': 'knowledge', 'mmlu_pro/history': 'knowledge',
        'supergpqa/Clinical Medicine': 'knowledge', 'supergpqa/Biology': 'knowledge'}


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    scores = [json.loads(line) for line in open(trapi.RUNS / args.run / 'scores.jsonl')]
    by_q = collections.defaultdict(dict)
    for s in scores:
        by_q[s['sample_id']][(s['model'], s['effort'])] = s
    complete = {q: c for q, c in by_q.items()
                if len(c) == len(CONFIGS) and all(s['score'] is not None for s in c.values())}
    print(f"{len(complete)} of {len(by_q)} questions have all {len(CONFIGS)} configurations scored\n")
    group_of = {q: next(iter(c.values()))['task_group'] for q, c in complete.items()}
    groups = sorted(set(group_of.values()))

    print(f"{'group':30s}" + ''.join(f"{m.replace('gpt-5', 'g5')[:7]:>7s}{e[0]}" for m, e in CONFIGS))
    for g in groups:
        qs = [q for q in complete if group_of[q] == g]
        print(f"{g:30s}" + ''.join(f"{mean([complete[q][c]['score'] for q in qs]):8.2f}" for c in CONFIGS))

    print('\nmean gain by task type (score points, over questions):')
    print(f"{'type':10s} {'n':>4s} {'effort hi-lo nano':>18s} {'mini':>6s} {'gpt-5':>6s} {'size g5-nano @low':>18s} {'@high':>6s}")
    kinds = collections.defaultdict(list)
    for q in complete:
        kinds[TYPE.get(group_of[q], 'livebench-other')].append(q)
    for kind, qs in sorted(kinds.items()):
        eff = [mean([complete[q][(m, 'high')]['score'] - complete[q][(m, 'low')]['score'] for q in qs]) for m in MODELS]
        size = [mean([complete[q][('gpt-5', e)]['score'] - complete[q][('gpt-5-nano', e)]['score'] for q in qs])
                for e in ('low', 'high')]
        print(f"{kind:10s} {len(qs):4d} {eff[0]:18.3f} {eff[1]:6.3f} {eff[2]:6.3f} {size[0]:18.3f} {size[1]:6.3f}")

    print('\noracle router vs fixed configurations, per dataset (score, $ per question):')
    for dataset in ('mmlu_pro', 'supergpqa', 'livebench', 'all'):
        qs = [q for q in complete if dataset == 'all' or q.startswith(dataset + ':')]
        fixed = {c: (mean([complete[q][c]['score'] for q in qs]), mean([complete[q][c]['cost_usd'] for q in qs]))
                 for c in CONFIGS}
        picks = []
        for q in qs:
            best = max(s['score'] for s in complete[q].values())
            picks.append(min((s for s in complete[q].values() if s['score'] == best), key=lambda s: s['cost_usd']))
        best_fixed = max(fixed, key=lambda c: fixed[c][0])
        cheap_ok = mean([complete[q][('gpt-5-nano', 'low')]['score'] == max(s['score'] for s in complete[q].values())
                         for q in qs])
        none = mean([max(s['score'] for s in complete[q].values()) == 0 for q in qs])
        print(f"  {dataset:9s} n={len(qs):3d}  oracle {mean([p['score'] for p in picks]):.3f} at ${mean([p['cost_usd'] for p in picks]):.5f}"
              f" | best fixed {best_fixed[0]}/{best_fixed[1]} {fixed[best_fixed][0]:.3f} at ${fixed[best_fixed][1]:.5f}"
              f" | gpt-5/high {fixed[('gpt-5', 'high')][0]:.3f} at ${fixed[('gpt-5', 'high')][1]:.5f}"
              f" | nano-low already best {cheap_ok:.0%}, none solve {none:.0%}")
        if dataset == 'all':
            counts = collections.Counter(f"{p['model']}/{p['effort']}" for p in picks)
            print('  oracle picks:', dict(counts.most_common()))


if __name__ == '__main__':
    main()
