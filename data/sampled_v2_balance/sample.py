"""Build setting-specific balanced training folds; original test folds stay intact."""
import collections
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 224
CONFIGS = [(m, e) for m in ('gpt-5-nano', 'gpt-5-mini', 'gpt-5')
           for e in ('low', 'medium', 'high')]
SETTINGS = [('quality', 1.0, 0.0), ('balanced', 0.5, 0.5), ('cost', 0.2, 0.8)]
QUOTAS = {'livebench': 100, 'supergpqa': 120, 'mmlu_pro': 100}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    source = ROOT / 'data/combined_v2_800.jsonl'
    score_path = ROOT / 'runs/full/scores.jsonl'
    questions, scores = read(source), read(score_path)
    lookup = {(s['sample_id'], s['model'], s['effort']): s for s in scores}
    assert len(lookup) == len(scores) == len(questions) * len(CONFIGS)
    manifest = {'seed': SEED, 'configs': CONFIGS,
                'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (source, score_path)}, 'splits': []}
    for name, alpha, beta in SETTINGS:
        for fold in range(5):
            train = [q for q in questions if q['fold'] != fold]
            test = [q for q in questions if q['fold'] == fold]
            responses = [lookup[q['sample_id'], m, e] for q in train for m, e in CONFIGS]
            p0, p1 = min(s['score'] for s in responses), max(s['score'] for s in responses)
            c0 = min(math.log(s['cost_usd']) for s in responses)
            c1 = max(math.log(s['cost_usd']) for s in responses)
            labels = {}
            for q in train:
                values = [lookup[q['sample_id'], m, e] for m, e in CONFIGS]
                rewards = [alpha * (s['score'] - p0) / (p1 - p0)
                           - beta * (math.log(s['cost_usd']) - c0) / (c1 - c0) for s in values]
                labels[q['sample_id']] = min(range(9), key=lambda k: (-rewards[k], values[k]['cost_usd'], k))
            removed = set()
            allocation = {}
            for dataset, quota in QUOTAS.items():
                groups = collections.defaultdict(list)
                for q in train:
                    if q['dataset'] == dataset:
                        groups[q['task_group']].append(q)
                candidates = {g: sorted(q['sample_id'] for q in rows if labels[q['sample_id']] == 0)
                              for g, rows in groups.items()}
                target = {g: len(rows) // 2 for g, rows in groups.items()}
                take = {g: min(target[g], len(candidates[g])) for g in groups}
                assert sum(map(len, candidates.values())) >= quota
                while sum(take.values()) < quota:
                    available = [g for g in groups if take[g] < len(candidates[g])]
                    g = min(available, key=lambda g: (take[g] - target[g], g))
                    take[g] += 1
                for g in sorted(groups):
                    rng = random.Random(f'{SEED}/{fold}/{g}')
                    removed.update(rng.sample(candidates[g], take[g]))
                    allocation[g] = {'before': len(groups[g]), 'nano_low_before': len(candidates[g]),
                                     'removed': take[g], 'after': len(groups[g]) - take[g]}
            retained = [q for q in train if q['sample_id'] not in removed]
            assert len(retained) == len(removed) == 320 and len(test) == 160
            assert all(labels[q] == 0 for q in removed)
            assert dict(collections.Counter(q['dataset'] for q in retained)) == QUOTAS
            assert not ({q['sample_id'] for q in retained} & {q['sample_id'] for q in test})
            folder = OUT / name / f'fold_{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            # Keep source question records intact; labels are separate and fixed before downsampling.
            write_rows(folder / 'train_balance.jsonl', retained)
            write_rows(folder / 'test.jsonl', test)
            split = {'setting': name, 'alpha': alpha, 'beta': beta, 'fold': fold,
                     'normalizer': {'score_min': p0, 'score_max': p1, 'log_cost_min': c0, 'log_cost_max': c1},
                     'groups': allocation, 'removed_ids': sorted(removed),
                     'train_gold_labels': {q['sample_id']: labels[q['sample_id']] for q in retained},
                     'label_counts_before': [sum(k == i for k in labels.values()) for i in range(9)],
                     'label_counts_after': [sum(labels[q['sample_id']] == i for q in retained) for i in range(9)],
                     'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (folder / 'train_balance.jsonl', folder / 'test.jsonl')}}
            (folder / 'manifest.json').write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n')
            manifest['splits'].append({'path': str(folder.relative_to(OUT)),
                                       'setting': name, 'fold': fold,
                                       'label_counts_after': split['label_counts_after']})
    (OUT / 'sampling_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Created 15 splits: each has 320 training questions and 160 unchanged test questions.')


if __name__ == '__main__':
    main()
