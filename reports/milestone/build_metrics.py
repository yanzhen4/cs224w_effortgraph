"""Recompute milestone tables from held-out predictions (Python standard library only)."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [(m, e) for m in ('gpt-5-nano', 'gpt-5-mini', 'gpt-5') for e in ('low', 'medium', 'high')]
SETTINGS = {'quality': (1, 0), 'balanced': (0.5, 0.5), 'cost': (0.2, 0.8)}


def read(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def main():
    questions = read('data/combined_v2_800.jsonl')
    scores = read('runs/full/scores.jsonl')
    folds = {q['sample_id']: q['fold'] for q in questions}
    lookup = {(s['sample_id'], s['model'], s['effort']): s for s in scores}
    assert len(questions) == len(folds) == 800 and len(lookup) == len(scores) == 7200
    normalizers = {}
    for fold in range(5):
        train = [s for s in scores if folds[s['sample_id']] != fold]
        normalizers[fold] = (min(s['score'] for s in train), max(s['score'] for s in train),
                             min(math.log(s['cost_usd']) for s in train), max(math.log(s['cost_usd']) for s in train))

    def metrics(picks, alpha, beta):
        assert set(picks) == set(folds)
        chosen = [lookup[q, *CONFIGS[picks[q]]] for q in folds]
        reward = []
        for s in chosen:
            p0, p1, c0, c1 = normalizers[folds[s['sample_id']]]
            reward.append(alpha * (s['score'] - p0) / (p1 - p0)
                          - beta * (math.log(s['cost_usd']) - c0) / (c1 - c0))
        return {'score': statistics.mean(s['score'] for s in chosen),
                'cost': statistics.mean(s['cost_usd'] for s in chosen), 'reward': statistics.mean(reward),
                'nano_low_share': sum(k == 0 for k in picks.values()) / 800}

    def collect(records):
        assert len(records) == 5 and {r['fold'] for r in records} == set(range(5))
        picks = {}
        for r in records:
            assert len(r['sample_ids']) == len(r['picks']) == 160
            assert all(folds[q] == r['fold'] and q not in picks for q in r['sample_ids'])
            picks.update(zip(r['sample_ids'], r['picks']))
        return picks

    result = {'seed': 0, 'questions': 800, 'settings': SETTINGS, 'methods': {}}
    for k, (m, e) in enumerate(CONFIGS):
        result['methods'][f'{m}/{e}'] = {s: metrics(dict.fromkeys(folds, k), a, b) for s, (a, b) in SETTINGS.items()}
    for name in ('mlp', 'mlp_balance'):
        records = read(f'runs/router/{name}/picks.jsonl')
        result['methods'][name] = {s: metrics(collect([r for r in records if r['seed'] == 0
                                                      and (r['alpha'], r['beta']) == (a, b)]), a, b)
                                   for s, (a, b) in SETTINGS.items()}
    for method in ('warm', 'scratch'):
        result['methods'][method] = {}
        for s, (a, b) in SETTINGS.items():
            folder = ROOT / 'runs/router/full_balance_50ep_seed0' / method / s
            records = [json.loads((folder / f'fold_{f}/result.json').read_text()) for f in range(5)]
            for f in range(5):
                cfg = json.loads((folder / f'fold_{f}/config.json').read_text())
                assert cfg['epochs'] == 50 and cfg['init'] == method and cfg['seed'] == 0
            result['methods'][method][s] = metrics(collect(records), a, b)
    oracle = {}
    for q in folds:
        oracle[q] = min(range(9), key=lambda k: (-lookup[q, *CONFIGS[k]]['score'],
                                                lookup[q, *CONFIGS[k]]['cost_usd'], k))
    result['methods']['quality_first_oracle'] = {s: metrics(oracle, a, b) for s, (a, b) in SETTINGS.items()}
    (Path(__file__).parent / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, values in result['methods'].items():
        print(name, ' '.join(f"{values[s]['reward']:.4f}" for s in SETTINGS))


if __name__ == '__main__':
    main()
