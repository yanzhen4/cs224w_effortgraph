"""Sample the v2 routing set: 15 task groups, one fixed stratified order per group, the first n of each order.

Each group's deduplicated pool is put in a seeded order that cycles through its strata (MMLU-Pro src, SuperGPQA
difficulty, LiveBench task and then native difficulty), so every prefix of the order is a stratified random sample.
The set is the first N[dataset] of each order; the pilot is the first PILOT; a later extension takes the next items.
Folds for 5-fold cross-validation are rank % 5, so they stay fixed when the set grows.
"""
import collections
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
SEED = 224
FOLDS = 5
PILOT = 10
N = {'livebench': 50, 'supergpqa': 60, 'mmlu_pro': 50}
GROUPS = {
    'livebench': ['math', 'reasoning', 'coding', 'data_analysis', 'instruction_following'],
    'supergpqa': ['Physics', 'Mathematics', 'Chemistry', 'Clinical Medicine', 'Biology'],
    'mmlu_pro': ['math', 'physics', 'health', 'law', 'history'],
}
SUPERGPQA_LEVELS = ['easy', 'middle', 'hard']


def cycle(strata, keys, rng_key):
    """Shuffle each stratum with its own seed, then take one item from each stratum in turn."""
    queues = []
    for key in keys:
        items = sorted(strata[key], key=lambda r: str(r['_id']))
        random.Random(f'{rng_key}/{key}').shuffle(items)
        queues.append(collections.deque(items))
    order = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.popleft())
    return order


def livebench_difficulty(row):
    value = {'coding': (row.get('original_json') or {}).get('difficulty'),
             'math': row.get('hardness'), 'reasoning': row.get('level')}.get(row['category'])
    return 'unknown' if value is None or value == '' else str(value)


def livebench_pools():
    for group in GROUPS['livebench']:
        source = ROOT / f'data/raw/livebench/{group}/jsonl/test-00000-of-00001.jsonl'
        rows = [json.loads(line) for line in source.open()]
        unique = {}
        # When identical prompts recur, retain the newest release deterministically (as in sampled/livebench).
        for row in sorted(rows, key=lambda r: (str(r.get('livebench_release_date') or ''), str(r['question_id'])),
                          reverse=True):
            unique.setdefault(json.dumps(row['turns'], ensure_ascii=False, sort_keys=True), {**row, '_id': row['question_id']})
        tasks = collections.defaultdict(list)
        for row in unique.values():
            tasks[row['task']].append(row)
        task_orders = {}
        for task, items in tasks.items():
            levels = collections.defaultdict(list)
            for row in items:
                levels[livebench_difficulty(row)].append(row)
            task_orders[task] = cycle(levels, sorted(levels), f'{SEED}/livebench/{group}/{task}')
        # Interleave tasks while keeping each task's own difficulty cycle.
        queues = [collections.deque(task_orders[t]) for t in sorted(task_orders)]
        order = []
        while any(queues):
            for queue in queues:
                if queue:
                    order.append(queue.popleft())
        yield group, source, len(rows), order, lambda r: f"{r['task']}/{livebench_difficulty(r)}"


def supergpqa_pools():
    source = ROOT / 'data/raw/supergpqa/SuperGPQA-all.jsonl'
    rows = [json.loads(line) for line in source.open()]
    for group in GROUPS['supergpqa']:
        strata = {}
        for level in SUPERGPQA_LEVELS:
            unique = {}
            for row in sorted((r for r in rows if r['field'] == group and r['difficulty'] == level), key=lambda r: r['uuid']):
                unique.setdefault(row['question'], {**row, '_id': row['uuid']})
            strata[level] = list(unique.values())
        raw = sum(1 for r in rows if r['field'] == group)
        yield group, source, raw, cycle(strata, SUPERGPQA_LEVELS, f'{SEED}/supergpqa/{group}'), lambda r: r['difficulty']


def mmlu_pro_pools():
    source = ROOT / 'data/raw/mmlu_pro/jsonl/test-00000-of-00001.jsonl'
    rows = [json.loads(line) for line in source.open()]
    for group in GROUPS['mmlu_pro']:
        unique = {}
        for row in sorted((r for r in rows if r['category'] == group), key=lambda r: r['question_id']):
            unique.setdefault(row['question'], {**row, '_id': row['question_id']})
        strata = collections.defaultdict(list)
        for row in unique.values():
            strata[row['src']].append(row)
        raw = sum(1 for r in rows if r['category'] == group)
        yield group, source, raw, cycle(strata, sorted(strata), f'{SEED}/mmlu_pro/{group}'), lambda r: r['src']


def main():
    output = ROOT / 'data/sampled_v2'
    manifest = {'seed': SEED, 'folds': FOLDS, 'pilot_per_group': PILOT, 'per_group': N,
                'policy': 'Deduplicated pools as in data/sampled; one seeded order per group cycling through strata; '
                          'the set is the first n of each order; fold = rank % 5; pilot = rank < 10.',
                'groups': {}}
    combined = []
    pools = {'livebench': livebench_pools, 'supergpqa': supergpqa_pools, 'mmlu_pro': mmlu_pro_pools}
    for dataset, pool_fn in pools.items():
        (output / dataset).mkdir(parents=True, exist_ok=True)
        (output / 'order').mkdir(parents=True, exist_ok=True)
        for group, source, raw, order, stratum in pool_fn():
            n = N[dataset]
            assert len(order) >= n, (dataset, group, len(order))
            assert len({str(r['_id']) for r in order}) == len(order)
            name = group.lower().replace(' ', '_')
            (output / 'order' / f'{dataset}__{name}.json').write_text(
                json.dumps([r['_id'] for r in order], ensure_ascii=False) + '\n')
            selected = [{k: v for k, v in r.items() if k != '_id'} for r in order[:n]]
            destination = output / dataset / f'{name}.jsonl'
            destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
            for rank, row in enumerate(selected):
                combined.append({'sample_id': f"{dataset}:{order[rank]['_id']}", 'dataset': dataset,
                                 'source_id': order[rank]['_id'], 'task_group': f'{dataset}/{group}',
                                 'rank': rank, 'fold': rank % FOLDS, 'pilot': rank < PILOT, 'record': row})
            manifest['groups'][f'{dataset}/{group}'] = {
                'raw_records': raw, 'pool_after_dedup': len(order), 'sampled': n,
                'strata_pool': dict(sorted(collections.Counter(stratum(r) for r in order).items())),
                'strata_sampled': dict(sorted(collections.Counter(stratum(r) for r in order[:n]).items())),
                'strata_pilot': dict(sorted(collections.Counter(stratum(r) for r in order[:PILOT]).items())),
                'source_file': str(source.relative_to(ROOT)), 'file': str(destination.relative_to(ROOT)),
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    total = sum(N[d] * len(g) for d, g in GROUPS.items())
    assert len(combined) == len({r['sample_id'] for r in combined}) == total
    destination = ROOT / f'data/combined_v2_{total}.jsonl'
    with destination.open('w') as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest['combined'] = {'file': str(destination.relative_to(ROOT)), 'records': total,
                            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    (output / 'sampling_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for group, info in manifest['groups'].items():
        print(f"{group:40s} pool {info['pool_after_dedup']:5d}  sampled {info['sampled']}  {info['strata_sampled']}")
    print(f'Wrote {total} questions ({sum(r["pilot"] for r in combined)} pilot) to {destination}')


if __name__ == '__main__':
    main()
