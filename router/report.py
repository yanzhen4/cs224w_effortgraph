"""Evaluate routers' test-fold picks against the fixed configurations and the non-learned references.

For each (alpha, beta) setting: per-data-set benchmark score, their macro average, the score over all 800 questions,
cost per question (USD, list prices) and routing overhead. A learned router is reported as the mean +- SD over seeds;
paired comparisons use each question's seed-averaged score, resampled over questions (10,000 bootstrap draws).
References: the 9 fixed configurations; random (the expectation over the 9); per-group best (each task group's
highest-mean-reward configuration on the training folds, no learning); oracle (each test question's highest-reward
configuration, an upper bound).

    python router/report.py --routers mlp
"""
import argparse
import collections
import json

import numpy as np

import data

NAMES = [f"{m}/{e}" for m, e in data.CONFIGS]


def summarize(t, s, c):
    per = {d: float(s[t.dataset == d].mean()) for d in data.DATASETS}
    return {**per, 'macro': float(np.mean(list(per.values()))), 'all': float(s.mean()), 'cost': float(c.mean())}


def chosen(t, picks):
    rows = np.arange(len(picks))
    return t.P[rows, picks], t.C[rows, picks]


def references(t, alpha, beta):
    """Test-fold picks of the per-group best and the oracle, each fold normalized on its own training questions."""
    group_best, oracle = np.zeros(len(t.ids), int), np.zeros(len(t.ids), int)
    for fold in range(data.FOLDS):
        train, test = t.fold != fold, t.fold == fold
        norm = data.Normalizer(t.P[train], t.C[train])
        picks = data.per_group_best(t, train, norm.reward(t.P[train], t.C[train], alpha, beta))
        group_best[test] = [picks[g] for g in t.group[test]]
        oracle[test] = data.best(norm.reward(t.P[test], t.C[test], alpha, beta), t.C[test])
    return group_best, oracle


def load_router(t, name):
    runs = collections.defaultdict(lambda: np.full(len(t.ids), -1))
    overhead = []
    index = {q: i for i, q in enumerate(t.ids)}
    for line in open(data.RUNS / 'router' / name / 'picks.jsonl'):
        r = json.loads(line)
        runs[(r['alpha'], r['beta'], r['seed'])][[index[q] for q in r['sample_ids']]] = r['picks']
        overhead.append(r['infer_ms_per_query'])
    assert all((p >= 0).all() for p in runs.values()), f'{name}: some test folds are missing'
    return runs, float(np.mean(overhead))


def paired(a, b, rng, draws=10_000):
    idx = rng.integers(0, len(a), (draws, len(a)))
    d = (a - b)[idx].mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt(m, sd=None):
    cells = [f"{m[k]:.3f}" + (f"±{sd[k]:.3f}" if sd else '') for k in (*data.DATASETS, 'macro', 'all')]
    return ' '.join(f"{c:>12s}" for c in cells) + f" {m['cost']:10.5f}" + (f"±{sd['cost']:.5f}" if sd else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--routers', nargs='+', default=['mlp'])
    args = ap.parse_args()
    t = data.load()
    emb = np.load(data.ROOT / 'data/embeddings/all-MiniLM-L6-v2.npz')
    rng = np.random.default_rng(0)
    header = f"{'':26s}" + ' '.join(f"{k:>12s}" for k in (*data.DATASETS, 'macro', 'all')) + f" {'$/question':>10s}"
    report = {'fixed': {}, 'settings': {}}

    print('fixed configurations (all 800 questions; the same under every setting)\n' + header)
    for k, name in enumerate(NAMES):
        m = summarize(t, *chosen(t, np.full(len(t.ids), k)))
        report['fixed'][name] = m
        print(f"{name:26s}" + fmt(m))
    m = summarize(t, t.P.mean(1), t.C.mean(1))
    report['fixed']['random'] = m
    print(f"{'random (expected)':26s}" + fmt(m))

    routers = {name: load_router(t, name) for name in args.routers}
    g5h = NAMES.index('gpt-5/high')
    for alpha, beta in data.SETTINGS:
        key = f"alpha={alpha},beta={beta}"
        group_best, oracle = references(t, alpha, beta)
        rows = {'per-group best': summarize(t, *chosen(t, group_best)), 'oracle': summarize(t, *chosen(t, oracle))}
        print(f"\n(alpha, beta) = ({alpha}, {beta})\n" + header)
        out = {'references': rows, 'routers': {}}
        for name, (runs, infer_ms) in routers.items():
            seeds = sorted(s for a, b, s in runs if (a, b) == (alpha, beta))
            per_seed = [summarize(t, *chosen(t, runs[(alpha, beta, s)])) for s in seeds]
            mean = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
            sd = {k: float(np.std([m[k] for m in per_seed], ddof=1)) for k in per_seed[0]}
            print(f"{name + f' ({len(seeds)} seeds)':26s}" + fmt(mean, sd))
            score = np.mean([chosen(t, runs[(alpha, beta, s)])[0] for s in seeds], axis=0)
            picks = collections.Counter(NAMES[k] for s in seeds for k in runs[(alpha, beta, s)])
            vs = {ref: paired(score, chosen(t, p)[0], rng) for ref, p in
                  (('per-group best', group_best), ('gpt-5/high', np.full(len(t.ids), g5h)))}
            out['routers'][name] = {'mean': mean, 'sd': sd, 'seeds': len(seeds), 'vs': vs,
                                    'pick_share': {k: v / sum(picks.values()) for k, v in picks.most_common()},
                                    'overhead_ms_per_query': {'embedding': float(emb['per_query_ms']),
                                                              'router': infer_ms}}
        for ref, m in rows.items():
            print(f"{ref:26s}" + fmt(m))
        for name, r in out['routers'].items():
            for ref, (d, lo, hi) in r['vs'].items():
                print(f"  {name} - {ref}: score (all 800) {d:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
            print(f"  {name} picks: " + ', '.join(f"{k} {v:.0%}" for k, v in r['pick_share'].items() if v >= 0.01))
        report['settings'][key] = out
    for name, r in out['routers'].items():
        o = r['overhead_ms_per_query']
        print(f"\nrouting overhead ({name}): embedding {o['embedding']:.1f} ms + router {o['router']:.3f} ms per query (CPU)")
    (data.RUNS / 'router' / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
